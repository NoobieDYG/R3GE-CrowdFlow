"use client"

import { useState, useRef, useEffect } from "react"
import { MessageSquare } from "lucide-react"

export default function AiChat() {
  const [messages, setMessages] = useState([])
  const [inputValue, setInputValue] = useState("")
  const [isTyping, setIsTyping] = useState(false)
  const messagesEndRef = useRef(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSendMessage = () => {
    if (inputValue.trim() === "") return

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: "user",
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsTyping(true)

    // Simulate AI response
    setTimeout(() => {
      const aiResponses = [
        "Zone 1 is currently at critical capacity. Consider redirecting people to Zone 3 which is at 6% capacity.",
        "Current crowd levels are within normal parameters in Zones 2, 3, and 4.",
        "Based on historical data, Zone 2 may reach capacity in approximately 45 minutes.",
        "I recommend opening additional entry points for Zone 1 to reduce congestion.",
      ]

      const randomResponse = aiResponses[Math.floor(Math.random() * aiResponses.length)]

      const aiMessage = {
        id: Date.now() + 1,
        text: randomResponse,
        sender: "ai",
      }

      setMessages((prev) => [...prev, aiMessage])
      setIsTyping(false)
    }, 1500)
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSendMessage()
    }
  }

  return (
    <div className="ai-chat-container">
      <div className="chat-header">
        <MessageSquare className="chat-icon" />
        <h2>AI Chat Interface</h2>
      </div>

      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="empty-chat">
            <p>Chat messages will appear here</p>
            <p className="chat-hint">Ask about crowd levels, zone statuses, or recommendations</p>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div key={message.id} className={`message ${message.sender === "user" ? "user-message" : "ai-message"}`}>
                {message.text}
              </div>
            ))}
            {isTyping && (
              <div className="message ai-message typing">
                <span className="typing-indicator">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="chat-input-container">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          className="chat-input"
        />
        <button onClick={handleSendMessage} className="send-button" disabled={inputValue.trim() === ""}>
          Send
        </button>
      </div>
    </div>
  )
}
