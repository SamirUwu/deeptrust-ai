"use client"

import { cn } from "@/lib/utils"
import { Shield, ShieldAlert, Volume2, MonitorPlay, Image, Cpu } from "lucide-react"

export interface AnalysisResult {
  type: "audio" | "video" | "image"
  probability: number
  confidence: "Low" | "Medium" | "High"
  label: "Authentic" | "Potential Deepfake"
  modelName?: string
}

interface AnalysisResultsProps {
  results: AnalysisResult[] | null
  isAnalyzing: boolean
}

function ResultCard({ result }: { result: AnalysisResult }) {
  const isAuthentic = result.label === "Authentic"
  const Icon = result.type === "audio" ? Volume2 : result.type === "video" ? MonitorPlay : Image
  const StatusIcon = isAuthentic ? Shield : ShieldAlert

  return (
    <div
      className={cn(
        "p-4 rounded-xl border transition-all",
        isAuthentic
          ? "bg-success/10 border-success/30"
          : "bg-destructive/10 border-destructive/30"
      )}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2">
          <div
            className={cn(
              "p-2 rounded-lg",
              isAuthentic ? "bg-success/20" : "bg-destructive/20"
            )}
          >
            <Icon
              className={cn(
                "h-4 w-4",
                isAuthentic ? "text-success" : "text-destructive"
              )}
            />
          </div>
          <span className="text-sm font-medium capitalize text-foreground">
            {result.type} Analysis
          </span>
        </div>
        <StatusIcon
          className={cn(
            "h-5 w-5",
            isAuthentic ? "text-success" : "text-destructive"
          )}
        />
      </div>

      <div className="space-y-3">
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-xs text-muted-foreground">Probability Score</span>
            <span className="text-sm font-mono font-semibold text-foreground">
              {result.probability.toFixed(2)}
            </span>
          </div>
          <div className="h-2 bg-secondary rounded-full overflow-hidden">
            <div
              className={cn(
                "h-full rounded-full transition-all duration-500",
                isAuthentic ? "bg-success" : "bg-destructive"
              )}
              style={{ width: `${result.probability * 100}%` }}
            />
          </div>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Confidence</span>
          <span
            className={cn(
              "text-xs font-medium px-2 py-0.5 rounded-full",
              result.confidence === "High" && "bg-success/20 text-success",
              result.confidence === "Medium" && "bg-warning/20 text-warning",
              result.confidence === "Low" && "bg-muted text-muted-foreground"
            )}
          >
            {result.confidence}
          </span>
        </div>

        <div className="flex items-center justify-between pt-2 border-t border-border/50">
          <span className="text-xs text-muted-foreground">Result</span>
          <span
            className={cn(
              "text-sm font-semibold",
              isAuthentic ? "text-success" : "text-destructive"
            )}
          >
            {result.label}
          </span>
        </div>

        {result.modelName && (
          <div className="flex items-center justify-between pt-2 border-t border-border/50">
            <span className="text-xs text-muted-foreground flex items-center gap-1">
              <Cpu className="h-3 w-3" />
              Model used
            </span>
            <span className="text-xs font-medium text-foreground">
              {result.modelName}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

export function AnalysisResults({ results, isAnalyzing }: AnalysisResultsProps) {
  if (isAnalyzing) {
    return (
      <div className="space-y-3">
        <label className="text-sm font-medium text-foreground">Analysis Results</label>
        <div className="grid gap-4 md:grid-cols-2">
          {[1, 2].map((i) => (
            <div
              key={i}
              className="p-4 rounded-xl border border-border bg-secondary/30 animate-pulse"
            >
              <div className="flex items-center gap-2 mb-4">
                <div className="h-8 w-8 rounded-lg bg-muted" />
                <div className="h-4 w-24 rounded bg-muted" />
              </div>
              <div className="space-y-3">
                <div className="h-2 rounded-full bg-muted" />
                <div className="h-4 w-16 rounded bg-muted ml-auto" />
                <div className="h-4 w-32 rounded bg-muted ml-auto" />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="space-y-3">
        <label className="text-sm font-medium text-foreground">Analysis Results</label>
        <div className="p-8 rounded-xl border border-dashed border-border text-center">
          <Shield className="h-10 w-10 mx-auto text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">
            Upload or record media and click Analyze to see results
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-foreground">Analysis Results</label>
      <div className="grid gap-4 md:grid-cols-2">
        {results.map((result, index) => (
          <ResultCard key={index} result={result} />
        ))}
      </div>
    </div>
  )
}