// simgpu-device-plugin is a fake Kubernetes device plugin. It registers a
// configurable number of simulated GPU devices with the kubelet, under a
// custom resource name (default simgpu.dev/gpu), so pods can request them
// like real GPUs and be scheduled, observed via `kubectl describe node`,
// and reasoned about with normal k8s resource requests/limits. No real
// hardware is touched — Allocate just hands the container an id and a
// couple of env vars describing the simulated part.
package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"time"

	"google.golang.org/grpc"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"
)

const kubeletSocket = pluginapi.KubeletSocket // /var/lib/kubelet/device-plugins/kubelet.sock

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

type simGpuServer struct {
	pluginapi.UnimplementedDevicePluginServer
	resourceName string
	gpuModel     string
	devices      []*pluginapi.Device
}

func newSimGpuServer(resourceName, gpuModel string, count int) *simGpuServer {
	devices := make([]*pluginapi.Device, 0, count)
	for i := 0; i < count; i++ {
		devices = append(devices, &pluginapi.Device{
			ID:     fmt.Sprintf("%s-%d", gpuModel, i),
			Health: pluginapi.Healthy,
		})
	}
	return &simGpuServer{resourceName: resourceName, gpuModel: gpuModel, devices: devices}
}

func (s *simGpuServer) GetDevicePluginOptions(context.Context, *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{}, nil
}

func (s *simGpuServer) ListAndWatch(_ *pluginapi.Empty, stream pluginapi.DevicePlugin_ListAndWatchServer) error {
	if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: s.devices}); err != nil {
		return err
	}
	// Simulated devices never change health; just keep the stream open.
	for {
		time.Sleep(30 * time.Second)
		if err := stream.Send(&pluginapi.ListAndWatchResponse{Devices: s.devices}); err != nil {
			return err
		}
	}
}

func (s *simGpuServer) Allocate(_ context.Context, req *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	resp := &pluginapi.AllocateResponse{}
	for _, r := range req.ContainerRequests {
		cResp := &pluginapi.ContainerAllocateResponse{
			Envs: map[string]string{
				"SIMGPU_MODEL":   s.gpuModel,
				"SIMGPU_DEVICES": fmt.Sprintf("%v", r.DevicesIDs),
			},
		}
		resp.ContainerResponses = append(resp.ContainerResponses, cResp)
	}
	return resp, nil
}

func (s *simGpuServer) PreStartContainer(context.Context, *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	return &pluginapi.PreStartContainerResponse{}, nil
}

func (s *simGpuServer) GetPreferredAllocation(context.Context, *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {
	return &pluginapi.PreferredAllocationResponse{}, nil
}

func serve(socketDir, socketName string, server *simGpuServer) (*grpc.Server, string, error) {
	socketPath := socketDir + "/" + socketName
	_ = os.Remove(socketPath)

	lis, err := net.Listen("unix", socketPath)
	if err != nil {
		return nil, "", fmt.Errorf("listen on %s: %w", socketPath, err)
	}
	grpcServer := grpc.NewServer()
	pluginapi.RegisterDevicePluginServer(grpcServer, server)
	go func() {
		if err := grpcServer.Serve(lis); err != nil {
			log.Fatalf("grpc serve failed: %v", err)
		}
	}()

	// Wait for the socket to be dialable before registering with kubelet.
	conn, err := dial(socketPath, 5*time.Second)
	if err != nil {
		return nil, "", fmt.Errorf("device plugin socket did not come up: %w", err)
	}
	conn.Close()

	return grpcServer, socketName, nil
}

func dial(unixSocketPath string, timeout time.Duration) (*grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	return grpc.DialContext(ctx, "unix://"+unixSocketPath, grpc.WithInsecure(), grpc.WithBlock()) //nolint:staticcheck
}

func register(kubeletSocket, socketName, resourceName string) error {
	conn, err := dial(kubeletSocket, 5*time.Second)
	if err != nil {
		return fmt.Errorf("dial kubelet: %w", err)
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)
	_, err = client.Register(context.Background(), &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     socketName,
		ResourceName: resourceName,
	})
	if err != nil {
		return fmt.Errorf("register with kubelet: %w", err)
	}
	return nil
}

func main() {
	socketDir := getenv("DEVICE_PLUGIN_DIR", "/var/lib/kubelet/device-plugins")
	resourceName := getenv("SIMGPU_RESOURCE_NAME", "simgpu.dev/gpu")
	gpuModel := getenv("SIMGPU_MODEL", "h100-sxm")
	count, err := strconv.Atoi(getenv("SIMGPU_COUNT", "8"))
	if err != nil {
		log.Fatalf("invalid SIMGPU_COUNT: %v", err)
	}

	server := newSimGpuServer(resourceName, gpuModel, count)
	socketName := "simgpu.sock"

	log.Printf("starting simgpu-device-plugin: resource=%s model=%s count=%d", resourceName, gpuModel, count)

	for {
		grpcServer, _, err := serve(socketDir, socketName, server)
		if err != nil {
			log.Printf("serve failed, retrying in 5s: %v", err)
			time.Sleep(5 * time.Second)
			continue
		}

		if err := register(socketDir+"/kubelet.sock", socketName, resourceName); err != nil {
			log.Printf("register failed, retrying in 5s: %v", err)
			grpcServer.Stop()
			time.Sleep(5 * time.Second)
			continue
		}

		log.Printf("registered %s with kubelet (%d simulated devices)", resourceName, count)
		select {} // serve forever; kubelet re-registration on restart is out of scope for this teaching tool
	}
}
