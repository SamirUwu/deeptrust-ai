"use client"

import { useState, useCallback, useEffect } from "react"
import { Scan, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Header } from "./header"
import { FileUpload } from "./file-upload"
import { RecordControls } from "./record-controls"
import { AnalysisResults, type AnalysisResult } from "./analysis-results"
import { AnalysisHistory, type HistoryItem } from "./analysis-history"
import { ModelSelector, type ModelOption, type FileCategory, getDefaultModel } from "./model-selector"
const API_URL = "http://localhost:8000"

function getFileCategory(file: File | null): FileCategory {
  if (!file) return null
  if (file.type.startsWith("video/")) return "video"
  if (file.type.startsWith("image/")) return "image"
  if (file.type.startsWith("audio/")) return "audio"
  return null
}

async function analyzeFile(file: File, model: ModelOption): Promise<AnalysisResult> {
  const formData = new FormData()
  formData.append("file", file)

  const response = await fetch(`${API_URL}${model.endpoint}`, {
    method: "POST",
    body: formData,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ error: "Analysis failed" }))
    throw new Error(error.error || "Analysis failed")
  }

  const data = await response.json()

  return {
    type: data.type || getFileCategory(file) || "video",
    probability: data.probability,
    confidence: data.confidence,
    label: data.label,
    modelName: model.name,
  }
}

export function DeepTrustDashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelOption | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult[] | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [error, setError] = useState<string | null>(null)

  const fileCategory = getFileCategory(selectedFile)

  // Reset model to default when file type changes
  useEffect(() => {
    if (fileCategory) {
      setSelectedModel(getDefaultModel(fileCategory))
    } else {
      setSelectedModel(null)
    }
  }, [fileCategory])

  const handleRecordingComplete = useCallback((file: File) => {
    setSelectedFile(file)
    setResults(null)
    setError(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile || !selectedModel) return

    setIsAnalyzing(true)
    setResults(null)
    setError(null)

    try {
      const analysisResult = await analyzeFile(selectedFile, selectedModel)
      setResults([analysisResult])

      const newHistoryItem: HistoryItem = {
        id: Date.now().toString(),
        fileName: selectedFile.name,
        fileType: fileCategory || "video",
        result: analysisResult.label === "Authentic" ? "Authentic" : "Potential Deepfake",
        timestamp: new Date(),
      }

      setHistory((prev) => [newHistoryItem, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed")
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedFile, selectedModel, fileCategory])

  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background pointer-events-none" />

      <main className="relative z-10 container max-w-4xl mx-auto px-4 py-8 md:py-12">
        <div className="space-y-8">
          <div className="bg-card/80 backdrop-blur-sm border border-border rounded-2xl shadow-xl shadow-primary/5 p-6 md:p-8 space-y-6">
            <Header />
            <FileUpload onFileSelect={setSelectedFile} selectedFile={selectedFile} />
            <RecordControls onRecordingComplete={handleRecordingComplete} />

            {/* Model Selector — aparece solo cuando hay archivo seleccionado */}
            <ModelSelector
              fileCategory={fileCategory}
              selectedModel={selectedModel}
              onModelSelect={setSelectedModel}
            />

            <Button
              size="lg"
              className="w-full h-14 text-lg font-semibold gap-2 bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:shadow-primary/40"
              onClick={handleAnalyze}
              disabled={!selectedFile || !selectedModel || isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Scan className="h-5 w-5" />
                  Analyze
                </>
              )}
            </Button>

            {error && (
              <div className="p-4 rounded-xl border border-destructive/30 bg-destructive/10 text-destructive text-sm">
                {error}
              </div>
            )}

            <AnalysisResults results={results} isAnalyzing={isAnalyzing} />
          </div>

          <div className="bg-card/80 backdrop-blur-sm border border-border rounded-2xl shadow-xl shadow-primary/5 p-6 md:p-8">
            <AnalysisHistory history={history} />
          </div>
        </div>

        <footer className="text-center mt-8 text-sm text-muted-foreground">
          <p>Powered by advanced AI models for media authenticity verification</p>
        </footer>
      </main>
    </div>
  )
}