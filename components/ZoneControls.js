"use client"

import { useState } from "react"

const initialZones = [
  { id: 1, name: "Zone 1", current: 110, capacity: 100, status: "CRITICAL" },
  { id: 2, name: "Zone 2", current: 26, capacity: 150, status: "NORMAL" },
  { id: 3, name: "Zone 3", current: 12, capacity: 200, status: "NORMAL" },
  { id: 4, name: "Zone 4", current: 17, capacity: 120, status: "NORMAL" },
]

export default function ZoneControls() {
  const [zones, setZones] = useState(initialZones)
  const [configuring, setConfiguring] = useState(null)

  const calculateStatus = (current, capacity) => {
    const percentage = (current / capacity) * 100
    if (percentage >= 90) return "CRITICAL"
    if (percentage >= 75) return "WARNING"
    return "NORMAL"
  }

  const getStatusColor = (status) => {
    switch (status) {
      case "CRITICAL":
        return "red"
      case "WARNING":
        return "orange"
      default:
        return "green"
    }
  }

  const getPercentage = (current, capacity) => {
    return Math.min(100, (current / capacity) * 100)
  }

  const handleConfigure = (zoneId) => {
    const zone = zones.find((z) => z.id === zoneId)
    setConfiguring(zone)
  }

  const handleCloseConfig = () => {
    setConfiguring(null)
  }

  const handleSaveConfig = (e) => {
    e.preventDefault()

    const current = Number.parseInt(e.target.current.value)
    const capacity = Number.parseInt(e.target.capacity.value)

    // Update the zone
    setZones(
      zones.map((zone) => {
        if (zone.id === configuring.id) {
          const status = calculateStatus(current, capacity)
          return { ...zone, current, capacity, status }
        }
        return zone
      }),
    )

    setConfiguring(null)
  }

  const handleIncrement = (zoneId, amount) => {
    setZones(
      zones.map((zone) => {
        if (zone.id === zoneId) {
          const newCurrent = Math.max(0, zone.current + amount)
          const status = calculateStatus(newCurrent, zone.capacity)
          return { ...zone, current: newCurrent, status }
        }
        return zone
      }),
    )
  }

  return (
    <div className="zone-controls-container">
      <h2 className="zone-header">Zone Controls</h2>

      <div className="zones-grid">
        {zones.map((zone) => (
          <div key={zone.id} className="zone-card" style={{ borderColor: getStatusColor(zone.status) }}>
            <h3 className="zone-title">{zone.name}</h3>

            <div className="progress-container">
              <div
                className="progress-bar"
                style={{
                  width: `${getPercentage(zone.current, zone.capacity)}%`,
                  backgroundColor: getStatusColor(zone.status),
                }}
              ></div>
            </div>

            <div className="zone-stats">
              {zone.current}/{zone.capacity}
            </div>

            <div className="zone-status">
              <span className="status-indicator" style={{ backgroundColor: getStatusColor(zone.status) }}></span>
              <span className="status-text">{zone.status}</span>
            </div>

            <div className="quick-controls">
              <button
                className="control-btn decrement"
                onClick={() => handleIncrement(zone.id, -1)}
                aria-label="Decrease occupancy"
              >
                -
              </button>
              <button className="configure-btn" onClick={() => handleConfigure(zone.id)}>
                Configure
              </button>
              <button
                className="control-btn increment"
                onClick={() => handleIncrement(zone.id, 1)}
                aria-label="Increase occupancy"
              >
                +
              </button>
            </div>
          </div>
        ))}
      </div>

      {configuring && (
        <div className="config-modal-overlay">
          <div className="config-modal">
            <h3>Configure {configuring.name}</h3>

            <form onSubmit={handleSaveConfig}>
              <div className="form-group">
                <label htmlFor="current">Current Occupancy:</label>
                <input type="number" id="current" name="current" min="0" defaultValue={configuring.current} />
              </div>

              <div className="form-group">
                <label htmlFor="capacity">Maximum Capacity:</label>
                <input type="number" id="capacity" name="capacity" min="1" defaultValue={configuring.capacity} />
              </div>

              <div className="modal-actions">
                <button type="button" className="cancel-btn" onClick={handleCloseConfig}>
                  Cancel
                </button>
                <button type="submit" className="save-btn">
                  Save Changes
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
