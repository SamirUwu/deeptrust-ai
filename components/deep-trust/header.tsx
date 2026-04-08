"use client"

import { ShieldCheck } from "lucide-react"

export function Header() {
  return (
    <header className="text-center space-y-3 pb-6 border-b border-border/50">
      <div className="flex items-center justify-center gap-3">
        <div className="relative">
          <div className="absolute inset-0 bg-primary/30 blur-xl rounded-full" />
          <div className="relative p-3 bg-primary/20 rounded-xl border border-primary/30">
            <ShieldCheck className="h-8 w-8 text-primary" />
          </div>
        </div>
        <h1 className="text-3xl font-bold tracking-tight text-foreground">
          DeepTrust AI
        </h1>
      </div>
      <p className="text-muted-foreground max-w-md mx-auto text-balance">
        Verify audio and video authenticity in seconds with advanced AI-powered deepfake detection
      </p>
    </header>
  )
}
