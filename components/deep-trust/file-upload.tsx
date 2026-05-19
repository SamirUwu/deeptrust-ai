"use client"

import { useState, useCallback } from "react"
import { Upload, FileAudio, FileVideo, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface FileUploadProps {
  onFileSelect: (file: File | null) => void
  selectedFile: File | null
}

export function FileUpload({ onFileSelect, selectedFile }: FileUploadProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file && (file.type.startsWith("audio/") || file.type.startsWith("video/"))) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      onFileSelect(file)
    }
  }, [onFileSelect])

  const clearFile = useCallback(() => {
    onFileSelect(null)
  }, [onFileSelect])

  const isAudio = selectedFile?.type.startsWith("audio/")
  const isVideo = selectedFile?.type.startsWith("video/")

  return (
    <div className="space-y-3">
      <label className="text-sm font-medium text-foreground">Upload File</label>
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          "relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 cursor-pointer",
          isDragging
            ? "border-primary bg-primary/10"
            : "border-border hover:border-primary/50 hover:bg-secondary/50",
          selectedFile && "border-primary/30 bg-secondary/30"
        )}
      >
        {!selectedFile && (
          <input
            type="file"
            accept="audio/*,video/*"
            onChange={handleFileChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
        )}
      
        {selectedFile ? (
          <div className="flex items-center justify-center gap-3">
            {isAudio && <FileAudio className="h-8 w-8 text-primary" />}
            {isVideo && <FileVideo className="h-8 w-8 text-primary" />}
            <div className="text-left">
              <p className="text-sm font-medium text-foreground truncate max-w-[200px]">
                {selectedFile.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation()
                clearFile()
              }}
              className="relative z-10 ml-2 p-1 rounded-full hover:bg-destructive/20 transition-colors"
            >
              <X className="h-4 w-4 text-destructive" />
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            <Upload className="h-10 w-10 mx-auto text-muted-foreground" />
            <p className="text-sm text-muted-foreground">
              Drag and drop your file here, or <span className="text-primary">browse</span>
            </p>
            <p className="text-xs text-muted-foreground/70">
              Supports audio, image and video files
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
