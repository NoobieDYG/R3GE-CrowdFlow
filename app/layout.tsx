import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Crowed Management',
  description: 'Created by R3GE',
  generator: 'R3GE',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
