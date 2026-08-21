{{/*
Resolves the K8s extended resource name for the currently selected
gpuBackend, so api.yaml/trainer-job.yaml never hardcode either backend's
resource name directly.
*/}}
{{- define "simgpu.gpuResourceName" -}}
{{- if eq .Values.gpuBackend "fake-gpu-operator" -}}
{{ .Values.fakeGpuOperator.resourceName }}
{{- else -}}
{{ .Values.devicePlugin.resourceName }}
{{- end -}}
{{- end -}}
