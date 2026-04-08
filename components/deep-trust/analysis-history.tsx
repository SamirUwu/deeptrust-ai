"use client"

import { cn } from "@/lib/utils"
import { FileAudio, FileVideo, Clock, Shield, ShieldAlert } from "lucide-react"

export interface HistoryItem {
  id: string
  fileName: string
  fileType: "audio" | "video"
  result: "Authentic" | "Potential Deepfake"
  timestamp: Date
}

interface AnalysisHistoryProps {
  history: HistoryItem[]
}

export function AnalysisHistory({ history }: AnalysisHistoryProps) {
  if (history.length === 0) {
    return (
      <div className="space-y-3">
        <label className="text-sm font-medium text-foreground">Analysis History</label>
        <div className="p-6 rounded-xl border border-dashed border-border text-center">
          <Clock className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
          <p className="text-sm text-muted-foreground">
            No analyses yet. Your history will appear here.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-foreground">Analysis History</label>
      <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2">
        {history.map((item) => {
          const isAuthentic = item.result === "Authentic"
          const FileIcon = item.fileType === "audio" ? FileAudio : FileVideo
          const StatusIcon = isAuthentic ? Shield : ShieldAlert

          return (
            <div
              key={item.id}
              className="flex items-center gap-3 p-3 rounded-lg bg-secondary/50 border border-border/50 hover:bg-secondary transition-colors"
            >
              <div
                className={cn(
                  "p-2 rounded-lg shrink-0",
                  isAuthentic ? "bg-success/20" : "bg-destructive/20"
                )}
              >
                <FileIcon
                  className={cn(
                    "h-4 w-4",
                    isAuthentic ? "text-success" : "text-destructive"
                  )}
                />
              </div>
              
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  {item.fileName}
                </p>
                <p className="text-xs text-muted-foreground">
                  {item.timestamp.toLocaleString()}
                </p>
              </div>
              
              <div className="flex items-center gap-2 shrink-0">
                <span
                  className={cn(
                    "text-xs font-medium px-2 py-1 rounded-full",
                    isAuthentic
                      ? "bg-success/20 text-success"
                      : "bg-destructive/20 text-destructive"
                  )}
                >
                  {item.result}
                </span>
                <StatusIcon
                  className={cn(
                    "h-4 w-4",
                    isAuthentic ? "text-success" : "text-destructive"
                  )}
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
