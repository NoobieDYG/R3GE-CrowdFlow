"use client"

import { useState, useEffect } from "react"
import { AlertCircle, Video } from "lucide-react"

export default function VideoFeed() {
  const [videoUrl, setVideoUrl] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(true)
  const [videoKey, setVideoKey] = useState(Date.now()) // Used to force re-render of video

  const handleLoadVideo = () => {
    if (!videoUrl.trim()) {
      setError(true)
      return
    }

    setIsLoading(true)
    setError(false)

    // Process the URL to ensure it works in an iframe
    let processedUrl = videoUrl

    // Handle YouTube URLs
    if (videoUrl.includes("youtube.com/watch?v=")) {
      const videoId = videoUrl.split("v=")[1]?.split("&")[0]
      if (videoId) {
        processedUrl = `https://www.youtube.com/embed/${videoId}`
      }
    } else if (videoUrl.includes("youtu.be/")) {
      const videoId = videoUrl.split("youtu.be/")[1]?.split("?")[0]
      if (videoId) {
        processedUrl = `https://www.youtube.com/embed/${videoId}`
      }
    }

    // Update the URL and reset the video component
    setVideoUrl(processedUrl)
    setVideoKey(Date.now())

    // Simulate loading
    setTimeout(() => {
      setIsLoading(false)
    }, 1000)
  }

  // Demo video for testing
  useEffect(() => {
    // Set a default demo video
    setVideoUrl("https://www.youtube.com/embed/dQw4w9WgXcQ")
    setError(false)
  }, [])

  return (
    <div className="video-feed-container">
      <div className="video-header">
        <Video className="video-icon" />
        <h2>Live Video Feed</h2>
      </div>

      <div className="video-player">
        {error ? (
          <div className="video-error">
            <AlertCircle size={48} />
            <p>An error occurred. Please try again later. (Playback ID: BaQAFZ7LH0CcdVqW)</p>
            <a href="#" className="learn-more">
              Learn More
            </a>
          </div>
        ) : isLoading ? (
          <div className="video-loading">
            <div className="spinner"></div>
            <p>Loading video...</p>
          </div>
        ) : (
          <iframe
            key={videoKey}
            src={videoUrl}
            title="Live Video Feed"
            allowFullScreen
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            className="video-iframe"
          ></iframe>
        )}
      </div>

      <div className="video-controls">
        <input
          type="text"
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
          placeholder="Enter YouTube or video URL"
          className="video-url-input"
        />
        <button onClick={handleLoadVideo} className="load-video-btn" disabled={isLoading}>
          {isLoading ? "Loading..." : "Load Video"}
        </button>
      </div>
    </div>
  )
}
