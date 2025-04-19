import VideoFeed from "@/components/VideoFeed"
import ZoneControls from "@/components/ZoneControls"
import AiChat from "@/components/AiChat"
import "./globals.css"

export default function CrowdManagementApp() {
  return (
    <div className="app-container">
      <div className="main-content">
        <div className="video-section">
          <VideoFeed />
        </div>
        <div className="chat-section">
          <AiChat />
        </div>
      </div>
      <div className="zones-section">
        <ZoneControls />
      </div>
    </div>
  )
}
