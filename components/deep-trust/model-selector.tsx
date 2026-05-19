"use client"

import { useEffect } from "react"
import { cn } from "@/lib/utils"
import { Cpu } from "lucide-react"

export type FileCategory = "video" | "image" | "audio" | null

export interface ModelOption {
  id: string
  name: string
  endpoint: string
  description?: string
  available: boolean
}

const VIDEO_MODELS: ModelOption[] = [
  { id: "efficientnet", name: "EfficientNet-B0", endpoint: "/api/analyze/video/efficientnet", description: "Fast and accurate", available: true },
  { id: "cnnlstm",      name: "CNN-LSTM",        endpoint: "/api/analyze/video/cnnlstm",      description: "Temporal analysis",  available: true },
]

const IMAGE_MODELS: ModelOption[] = [
  { id: "face",    name: "Face Detector",  endpoint: "/api/analyze/image/face",    description: "Optimized for human faces",      available: true },
  { id: "general", name: "Image Detector", endpoint: "/api/analyze/image/general", description: "Objects, animals, scenes",        available: true },
]

const AUDIO_MODEL: ModelOption = {
  id: "wav2vec2",
  name: "Wav2Vec2",
  endpoint: "/api/analyze/audio",
  description: "Audio waveform analysis",
  available: true,
}

export function getModelsForCategory(category: FileCategory): ModelOption[] {
  switch (category) {
    case "video":
      return VIDEO_MODELS
    case "image":
      return IMAGE_MODELS
    case "audio":
      return [AUDIO_MODEL]
    default:
      return []
  }
}

export function getDefaultModel(category: FileCategory): ModelOption | null {
  const models = getModelsForCategory(category)
  return models[0] || null
}

interface ModelSelectorProps {
  fileCategory: FileCategory
  selectedModel: ModelOption | null
  onModelSelect: (model: ModelOption) => void
}

export function ModelSelector({ fileCategory, selectedModel, onModelSelect }: ModelSelectorProps) {
  const models = getModelsForCategory(fileCategory)

  // Reset to default when category changes
  useEffect(() => {
    if (fileCategory) {
      const defaultModel = getDefaultModel(fileCategory)
      if (defaultModel && (!selectedModel || !models.find(m => m.id === selectedModel.id))) {
        onModelSelect(defaultModel)
      }
    }
  }, [fileCategory, models, selectedModel, onModelSelect])

  if (!fileCategory) {
    return null
  }

  // Audio has only one model - show label instead of dropdown
  if (fileCategory === "audio") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-foreground">Analysis Model</label>
        <div className="flex items-center gap-3 p-3 rounded-xl border border-border bg-secondary/30">
          <div className="p-2 rounded-lg bg-primary/20">
            <Cpu className="h-4 w-4 text-primary" />
          </div>
          <div>
            <p className="text-sm font-medium text-foreground">{AUDIO_MODEL.name}</p>
            <p className="text-xs text-muted-foreground">{AUDIO_MODEL.description}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-foreground">Select Analysis Model</label>
      <div className="grid gap-2">
        {models.map((model) => (
          <button
            key={model.id}
            type="button"
            onClick={() => onModelSelect(model)}
            className={cn(
              "flex items-center gap-3 p-3 rounded-xl border transition-all text-left",
              selectedModel?.id === model.id
                ? "border-primary bg-primary/10"
                : "border-border bg-secondary/30 hover:border-primary/50 hover:bg-secondary/50"
            )}
          >
            <div
              className={cn(
                "p-2 rounded-lg",
                selectedModel?.id === model.id ? "bg-primary/20" : "bg-muted"
              )}
            >
              <Cpu
                className={cn(
                  "h-4 w-4",
                  selectedModel?.id === model.id ? "text-primary" : "text-muted-foreground"
                )}
              />
            </div>
            <div>
              <p className="text-sm font-medium text-foreground">{model.name}</p>
              {model.description && (
                <p className="text-xs text-muted-foreground">{model.description}</p>
              )}
            </div>
            {selectedModel?.id === model.id && (
              <div className="ml-auto h-2 w-2 rounded-full bg-primary" />
            )}
          </button>
        ))}
      </div>
    </div>
  )
}
