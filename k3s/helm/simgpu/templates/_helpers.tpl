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

{{/*
Resolves one component's full image reference from the shared `image:`
defaults plus that component's own `<component>.image` block, e.g.:
  {{ include "simgpu.image" (dict "root" $ "component" .Values.api.image) }}
Precedence per field: component override -> shared `image.<field>` default.
`repository` is prefixed with `image.registry` only when it's a single path
segment (e.g. "api", matching this chart's own published GHCR layout, one
repo per component under image.registry). Any repository already containing
a "/" — a local dev tag like "simgpu/api", or a full third-party reference
like "myregistry.example.com/api" — is used as-is, since it's already fully
qualified and doesn't want image.registry prepended.
*/}}
{{- define "simgpu.image" -}}
{{- $root := .root -}}
{{- $img := .component -}}
{{- $repo := $img.repository -}}
{{- $tag := $img.tag | default $root.Values.image.tag -}}
{{- $registry := $root.Values.image.registry -}}
{{- if or (contains "/" $repo) (not $registry) -}}
{{- $repo -}}
{{- else -}}
{{- printf "%s/%s" $registry $repo -}}
{{- end -}}
:{{ $tag }}
{{- end -}}

{{/*
Resolves one component's imagePullPolicy: component override, else the
shared `image.pullPolicy` default.
*/}}
{{- define "simgpu.imagePullPolicy" -}}
{{- .component.pullPolicy | default .root.Values.image.pullPolicy -}}
{{- end -}}
