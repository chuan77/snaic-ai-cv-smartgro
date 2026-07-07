import { useEffect } from 'react'
import { CameraFeed } from '@/components/CameraFeed'
import { CartSidebar } from '@/components/CartSidebar'
import { HeaderBar } from '@/components/HeaderBar'
import { SectionHeader } from '@/components/SectionHeader'
import { useSmartCart } from '@/hooks/useSmartCart'

function App() {
  const loadCatalog = useSmartCart((state) => state.loadCatalog)

  useEffect(() => {
    void loadCatalog()
  }, [loadCatalog])

  return (
    <div className="min-h-screen bg-canvas p-6">
      <HeaderBar />
      <main className="mt-6 flex flex-col gap-6 lg:flex-row">
        <div className="flex-1">
          <SectionHeader index="01" title="Live Detection" />
          <CameraFeed />
        </div>
        <CartSidebar />
      </main>
    </div>
  )
}

export default App
