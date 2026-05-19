"use client"

import { useState, useRef, useCallback } from "react"
import { Mic, Video, Square, Camera } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface RecordControlsProps {
  onRecordingComplete: (file: File) => void
}

export function RecordControls({ onRecordingComplete }: RecordControlsProps) {
  const [isRecordingAudio, setIsRecordingAudio] = useState(false)
  const [isRecordingVideo, setIsRecordingVideo] = useState(false)
  const [isCapturingPhoto, setIsCapturingPhoto] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef        = useRef<MediaStream | null>(null)
  const chunksRef        = useRef<Blob[]>([])
  const timerRef         = useRef<NodeJS.Timeout | null>(null)

  const startTimer = useCallback(() => {
    setRecordingTime(0)
    timerRef.current = setInterval(() => {
      setRecordingTime((prev) => prev + 1)
    }, 1000)
  }, [])

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    setRecordingTime(0)
  }, [])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`
  }

  const startRecording = useCallback(async (type: "audio" | "video") => {
    try {
      const constraints: MediaStreamConstraints =
        type === "audio" ? { audio: true } : { audio: true, video: true }

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream

      const mimeType      = type === "audio" ? "audio/webm" : "video/webm"
      const mediaRecorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = mediaRecorder
      chunksRef.current = []

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = () => {
        const blob     = new Blob(chunksRef.current, { type: mimeType })
        const fileName = `recorded_${type}_${Date.now()}.webm`
        const file     = new File([blob], fileName, { type: mimeType })
        onRecordingComplete(file)

        if (streamRef.current) {
          streamRef.current.getTracks().forEach((track) => track.stop())
          streamRef.current = null
        }
      }

      mediaRecorder.start()
      startTimer()

      if (type === "audio") setIsRecordingAudio(true)
      else setIsRecordingVideo(true)
    } catch (error) {
      console.error("Error accessing media devices:", error)
      alert("Could not access your microphone/camera. Please check permissions.")
    }
  }, [onRecordingComplete, startTimer])

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop()
    }
    stopTimer()
    setIsRecordingAudio(false)
    setIsRecordingVideo(false)
  }, [stopTimer])

  const handleAudioClick = useCallback(() => {
    if (isRecordingAudio) stopRecording()
    else { if (isRecordingVideo) stopRecording(); startRecording("audio") }
  }, [isRecordingAudio, isRecordingVideo, startRecording, stopRecording])

  const handleVideoClick = useCallback(() => {
    if (isRecordingVideo) stopRecording()
    else { if (isRecordingAudio) stopRecording(); startRecording("video") }
  }, [isRecordingAudio, isRecordingVideo, startRecording, stopRecording])

  const handleTakePhoto = useCallback(async () => {
    try {
      setIsCapturingPhoto(true)

      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      streamRef.current = stream

      const video = document.createElement("video")
      video.srcObject = stream
      video.setAttribute("playsinline", "true")
      await video.play()

      await new Promise((resolve) => setTimeout(resolve, 500))

      const canvas    = document.createElement("canvas")
      canvas.width    = video.videoWidth
      canvas.height   = video.videoHeight
      const ctx       = canvas.getContext("2d")

      if (ctx) {
        ctx.drawImage(video, 0, 0)
        canvas.toBlob((blob) => {
          if (blob) {
            const fileName = `captured_photo_${Date.now()}.jpg`
            const file     = new File([blob], fileName, { type: "image/jpeg" })
            onRecordingComplete(file)
          }
          stream.getTracks().forEach((track) => track.stop())
          streamRef.current = null
          setIsCapturingPhoto(false)
        }, "image/jpeg", 0.9)
      }
    } catch (error) {
      console.error("Error capturing photo:", error)
      alert("Could not access your camera. Please check permissions.")
      setIsCapturingPhoto(false)
    }
  }, [onRecordingComplete])

  const isRecording = isRecordingAudio || isRecordingVideo

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-foreground">Record from Browser</label>

      {isRecording && (
        <div className="flex items-center justify-center gap-2 py-2">
          <span className="h-3 w-3 rounded-full bg-destructive animate-pulse" />
          <span className="text-sm font-mono text-foreground">{formatTime(recordingTime)}</span>
        </div>
      )}

      <div className="flex gap-3">
        <Button
          variant={isRecordingAudio ? "destructive" : "secondary"}
          className={cn("flex-1 h-12 gap-2 transition-all", isRecordingAudio && "animate-pulse")}
          onClick={handleAudioClick}
          disabled={isRecordingVideo}
        >
          {isRecordingAudio ? (
            <><Square className="h-4 w-4 fill-current" />Stop Audio</>
          ) : (
            <><Mic className="h-4 w-4" />Record Audio</>
          )}
        </Button>

        <Button
          variant={isRecordingVideo ? "destructive" : "secondary"}
          className={cn("flex-1 h-12 gap-2 transition-all", isRecordingVideo && "animate-pulse")}
          onClick={handleVideoClick}
          disabled={isRecordingAudio}
        >
          {isRecordingVideo ? (
            <><Square className="h-4 w-4 fill-current" />Stop Video</>
          ) : (
            <><Video className="h-4 w-4" />Record Video</>
          )}
        </Button>
      </div>

      <div className="flex justify-center">
        <Button
          variant="secondary"
          className={cn("h-12 w-64 gap-2 transition-all", isCapturingPhoto && "animate-pulse")}
          onClick={handleTakePhoto}
          disabled={isRecording || isCapturingPhoto}
        >
          <Camera className="h-4 w-4" />
          {isCapturingPhoto ? "Capturing..." : "Take Photo"}
        </Button>
      </div>

      <p className="text-xs text-muted-foreground text-center">
        {isRecording
          ? "Recording in progress... Click stop when finished"
          : "Click to start recording from your device"}
      </p>
    </div>
  )
}