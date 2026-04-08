"use client"

import { useState, useCallback } from "react"
import { Scan, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Header } from "./header"
import { FileUpload } from "./file-upload"
import { RecordControls } from "./record-controls"
import { AnalysisResults, type AnalysisResult } from "./analysis-results"
import { AnalysisHistory, type HistoryItem } from "./analysis-history"

// Mock analysis function - simulates AI analysis
function simulateAnalysis(): Promise<AnalysisResult[]> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const audioAuthentic = Math.random() > 0.4
      const videoAuthentic = Math.random() > 0.4
      
      resolve([
        {
          type: "audio",
          probability: audioAuthentic ? 0.85 + Math.random() * 0.12 : 0.15 + Math.random() * 0.3,
          confidence: audioAuthentic ? "High" : "Medium",
          label: audioAuthentic ? "Authentic" : "Potential Deepfake",
        },
        {
          type: "video",
          probability: videoAuthentic ? 0.82 + Math.random() * 0.15 : 0.12 + Math.random() * 0.28,
          confidence: videoAuthentic ? "High" : "Medium",
          label: videoAuthentic ? "Authentic" : "Potential Deepfake",
        },
      ])
    }, 2500)
  })
}

export function DeepTrustDashboard() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<AnalysisResult[] | null>(null)
  const [history, setHistory] = useState<HistoryItem[]>([])

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setResults(null)

    try {
      const analysisResults = await simulateAnalysis()
      setResults(analysisResults)

      // Add to history
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
    } finally {
      setIsAnalyzing(false)
    }
  }, [selectedFile])

  return (
    <div className="min-h-screen bg-background">
      {/* Background gradient effect */}
      <div className="fixed inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background pointer-events-none" />
      
      <main className="relative z-10 container max-w-4xl mx-auto px-4 py-8 md:py-12">
        <div className="space-y-8">
          {/* Main Card */}
          <div className="bg-card/80 backdrop-blur-sm border border-border rounded-2xl shadow-xl shadow-primary/5 p-6 md:p-8 space-y-6">
            <Header />

            {/* Upload Section */}
            <FileUpload onFileSelect={setSelectedFile} selectedFile={selectedFile} />

            {/* Record Section */}
            <RecordControls />

            {/* Analyze Button */}
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

            {/* Results Section */}
            <AnalysisResults results={results} isAnalyzing={isAnalyzing} />
          </div>

          {/* History Card */}
          <div className="bg-card/80 backdrop-blur-sm border border-border rounded-2xl shadow-xl shadow-primary/5 p-6 md:p-8">
            <AnalysisHistory history={history} />
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-8 text-sm text-muted-foreground">
          <p>Powered by advanced AI models for media authenticity verification</p>
        </footer>
      </main>
    </div>
  )
}
