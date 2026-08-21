FROM golang:1.23-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
RUN CGO_ENABLED=0 go build -o /out/simgpu-device-plugin .

FROM gcr.io/distroless/static-debian12
COPY --from=build /out/simgpu-device-plugin /simgpu-device-plugin
ENTRYPOINT ["/simgpu-device-plugin"]
