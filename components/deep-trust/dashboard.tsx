"use client"

import { useState, useCallback } from "react"
import { Scan, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Header } from "./header"
import { FileUpload } from "./file-upload"
import { RecordControls } from "./record-controls"
import { AnalysisResults, type AnalysisResult } from "./analysis-results"
import { AnalysisHistory, type HistoryItem } from "./analysis-history"

const API_URL = "http://localhost:8000"

async function analyzeFile(file: File): Promise<AnalysisResult[]> {
  const isAudio = file.type.startsWith("audio/")
  const isVideo = file.type.startsWith("video/")
  const isImage = file.type.startsWith("image/")

  if (!isAudio && !isVideo && !isImage) {
    throw new Error("Unsupported file type")
  }

  const formData = new FormData()
  formData.append("file", file)

  if (isAudio) {
    const res = await fetch(`${API_URL}/api/analyze/audio`, {
      method: "POST", body: formData,
    })
    if (!res.ok) throw new Error("Audio analysis failed")
    return [await res.json()]
  }

  if (isVideo) {
    const res = await fetch(`${API_URL}/api/analyze/video`, {
      method: "POST", body: formData,
    })
    if (!res.ok) throw new Error("Video analysis failed")
    return [await res.json()]
  }

  if (isImage) {
    const res = await fetch(`${API_URL}/api/analyze/image`, {
      method: "POST", body: formData,
    })
    if (!res.ok) throw new Error("Image analysis failed")
    return [await res.json()]
  }

  return []
}

export function DeepTrustDashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult[] | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleRecordingComplete = useCallback((file: File) => {
    setSelectedFile(file)
    setResults(null)
    setError(null)
  }, [])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setResults(null)
    setError(null)

    try {
      const analysisResults = await analyzeFile(selectedFile)
      setResults(analysisResults)

      const overallResult = analysisResults.every((r) => r.label === "Authentic")
        ? "Authentic"
        : "Potential Deepfake"

      const newHistoryItem: HistoryItem = {
        id: Date.now().toString(),
        fileName: selectedFile.name,
        fileType: selectedFile.type.startsWith("audio/") ? "audio" : "video",
        result: overallResult,
        timestamp: new Date(),
      }

      setHistory((prev) => [newHistoryItem, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed")
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedFile])

  return (
    <div className="min-h-screen bg-background">
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background pointer-events-none" />

      <main className="relative z-10 container max-w-4xl mx-auto px-4 py-8 md:py-12">
        <div className="space-y-8">
          <div className="bg-card/80 backdrop-blur-sm border border-border rounded-2xl shadow-xl shadow-primary/5 p-6 md:p-8 space-y-6">
            <Header />
            <FileUpload onFileSelect={setSelectedFile} selectedFile={selectedFile} />
            <RecordControls onRecordingComplete={handleRecordingComplete} />

            <Button
              size="lg"
              className="w-full h-14 text-lg font-semibold gap-2 bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/25 transition-all hover:shadow-primary/40"
              onClick={handleAnalyze}
              disabled={!selectedFile || isAnalyzing}
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