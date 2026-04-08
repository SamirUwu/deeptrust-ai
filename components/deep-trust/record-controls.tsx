"use client"

import { useState } from "react"
import { Mic, Video, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export function RecordControls() {
  const [isRecordingAudio, setIsRecordingAudio] = useState(false)
  const [isRecordingVideo, setIsRecordingVideo] = useState(false)

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-foreground">Record from Browser</label>
      <div className="flex gap-3">
        <Button
          variant={isRecordingAudio ? "destructive" : "secondary"}
          className={cn(
            "flex-1 h-12 gap-2 transition-all",
            isRecordingAudio && "animate-pulse"
          )}
          onClick={() => setIsRecordingAudio(!isRecordingAudio)}
        >
          {isRecordingAudio ? (
            <>
              <Square className="h-4 w-4 fill-current" />
              Stop Audio
            </>
          ) : (
            <>
              <Mic className="h-4 w-4" />
              Record Audio
            </>
          )}
        </Button>
        
        <Button
          variant={isRecordingVideo ? "destructive" : "secondary"}
          className={cn(
            "flex-1 h-12 gap-2 transition-all",
            isRecordingVideo && "animate-pulse"
          )}
          onClick={() => setIsRecordingVideo(!isRecordingVideo)}
        >
          {isRecordingVideo ? (
            <>
              <Square className="h-4 w-4 fill-current" />
              Stop Video
            </>
          ) : (
            <>
              <Video className="h-4 w-4" />
              Record Video
            </>
          )}
        </Button>
      </div>
      <p className="text-xs text-muted-foreground text-center">
        Click to start recording from your device
      </p>
    </div>
  )
}
