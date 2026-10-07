import { useState, useEffect, useRef, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Html } from '@react-three/drei'
import type { RunResult, PlacedBox, Container } from '../types/api'
import { UnplacedCartons } from './UnplacedCartons'

interface LoadingPlanViewerProps {
  result: RunResult
  onNewRun: () => void
}

const COLORS: Record<number, string> = {
  1: '#3b82f6',
  2: '#10b981',
  3: '#f59e0b',
  4: '#8b5cf6',
  5: '#ec4899',
  6: '#06b6d4',
  7: '#ef4444',
  8: '#84cc16',
}

/**
 * Coordinate System Agreement (Logistics → Three.js):
 *
 * Logistics Coordinate System:
 *   Origin (0,0,0) = Rear Wall - Bottom (Floor) - Left Wall
 *   X ∈ [0, L] – Length along container from rear wall (X = 0) to door (X = L)
 *   Y ∈ [0, W] – Width from left wall (Y = 0) to right wall (Y = W)
 *   Z ∈ [0, H] – Height vertically above floor (Z = 0) to ceiling (Z = H)
 *
 * Three.js World System:
 *   X_three = box.x + box.actual_length / 2 - L / 2   (Horizontal axis, centered at 0: Rear Wall = -L/2, Door = +L/2)
 *   Y_three = box.z + box.actual_height / 2           (Vertical UP axis; floor rests at Y = 0)
 *   Z_three = box.y + box.actual_width / 2 - W / 2    (Depth axis, centered at 0: Left Wall = -W/2, Right Wall = +W/2)
 *
 * Geometry: <boxGeometry args={[box.actual_length, box.actual_height, box.actual_width]} />
 */

function ContainerWireframe({ container }: { container: Container }) {
  const L = container.internal_length_cm
  const W = container.internal_width_cm
  const H = container.internal_height_cm

  return (
    <group>
      {/* Container wireframe box – centered at [0, H/2, 0] so floor rests at Y=0 */}
      <mesh position={[0, H / 2, 0]}>
        <boxGeometry args={[L, H, W]} />
        <meshBasicMaterial color="#64748b" wireframe transparent opacity={0.35} />
      </mesh>

      {/* Rear Wall indicator panel at X = -L/2 */}
      <Html position={[-L / 2 - 10, H + 12, 0]} center>
        <div style={{ fontSize: 11, color: '#64748b', fontWeight: 600, whiteSpace: 'nowrap' }}>
          REAR WALL
        </div>
      </Html>

      {/* Door indicator panel at X = L/2 (door end) */}
      <mesh position={[L / 2, H / 2, 0]} rotation={[0, Math.PI / 2, 0]}>
        <planeGeometry args={[W, H]} />
        <meshBasicMaterial color="#3b82f6" transparent opacity={0.08} side={2} />
      </mesh>
    </group>
  )
}

function BoxMesh({
  box,
  containerL,
  containerW,
  onClick,
  onHover,
  isSelected,
  isHovered,
  isCurrentStep,
}: {
  box: PlacedBox
  containerL: number
  containerW: number
  onClick: () => void
  onHover: (hovered: boolean) => void
  isSelected: boolean
  isHovered: boolean
  isCurrentStep: boolean
}) {
  const defaultColor = box.color || COLORS[box.customer_sequence % 8 || 8] || '#3b82f6'

  // Logistics → Three.js coordinate transform
  const posX = box.x + box.actual_length / 2 - containerL / 2
  const posY = box.z + box.actual_height / 2
  const posZ = box.y + box.actual_width / 2 - containerW / 2

  const boxColor = isSelected ? '#f59e0b' : isCurrentStep ? '#e11d48' : defaultColor
  const opacity = isSelected ? 0.95 : isHovered ? 0.92 : isCurrentStep ? 0.95 : 0.85

  return (
    <group position={[posX, posY, posZ]}>
      <mesh
        onClick={(e) => { e.stopPropagation(); onClick() }}
        onPointerOver={(e) => { e.stopPropagation(); onHover(true) }}
        onPointerOut={(e) => { e.stopPropagation(); onHover(false) }}
      >
        <boxGeometry args={[box.actual_length, box.actual_height, box.actual_width]} />
        <meshStandardMaterial
          color={boxColor}
          transparent
          opacity={opacity}
          metalness={0.1}
          roughness={0.7}
        />
      </mesh>
      {/* Box outline edges for clean distinction */}
      <lineSegments>
        <edgesGeometry args={[new (THREE_BOX_GEOMETRY as any)(box.actual_length, box.actual_height, box.actual_width)]} />
        <lineBasicMaterial color={isSelected || isCurrentStep ? '#ffffff' : '#0f172a'} transparent opacity={0.25} />
      </lineSegments>
    </group>
  )
}

// Helper reference for edges geometry constructor
import * as THREE from 'three'
const THREE_BOX_GEOMETRY = THREE.BoxGeometry

function CogMarker({ metrics, containerL, containerW }: {
  metrics: RunResult['metrics']
  containerL: number
  containerW: number
}) {
  const cogX = metrics.cog_x - containerL / 2
  const cogY = metrics.cog_z
  const cogZ = metrics.cog_y - containerW / 2

  return (
    <group position={[cogX, cogY, cogZ]}>
      <mesh>
        <sphereGeometry args={[7, 16, 16]} />
        <meshStandardMaterial color="#ef4444" transparent opacity={0.8} emissive="#ef4444" emissiveIntensity={0.4} />
      </mesh>
      {[
        { start: [-12, 0, 0], end: [12, 0, 0] },
        { start: [0, -12, 0], end: [0, 12, 0] },
        { start: [0, 0, -12], end: [0, 0, 12] },
      ].map((line, i) => (
        <line key={i}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([...line.start, ...line.end])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#ef4444" linewidth={2} />
        </line>
      ))}
      <Html center distanceFactor={400}>
        <div style={{
          fontSize: 10,
          color: '#ef4444',
          fontWeight: 700,
          background: 'rgba(255,255,255,0.9)',
          padding: '1px 5px',
          borderRadius: 3,
          border: '1px solid #fca5a5',
          whiteSpace: 'nowrap',
        }}>
          CoG
        </div>
      </Html>
    </group>
  )
}

function Scene({
  placedBoxes,
  container,
  metrics,
  selectedBoxId,
  hoveredBoxId,
  currentStep,
  onBoxClick,
  onBoxHover,
}: {
  placedBoxes: PlacedBox[]
  container: Container
  metrics: RunResult['metrics']
  selectedBoxId: string | null
  hoveredBoxId: string | null
  currentStep: number
  onBoxClick: (box: PlacedBox) => void
  onBoxHover: (box: PlacedBox | null) => void
}) {
  const L = container.internal_length_cm
  const W = container.internal_width_cm
  const H = container.internal_height_cm

  return (
    <>
      <ambientLight intensity={0.85} />
      <directionalLight position={[L * 1.2, H * 2.5, W * 1.5]} intensity={0.8} />
      <directionalLight position={[-L * 1.2, H * 1.5, -W * 1.5]} intensity={0.35} />

      <ContainerWireframe container={container} />

      {placedBoxes.map((box) => (
        <BoxMesh
          key={box.box_id}
          box={box}
          containerL={L}
          containerW={W}
          isSelected={box.box_id === selectedBoxId}
          isHovered={box.box_id === hoveredBoxId}
          isCurrentStep={box.step_index === currentStep}
          onClick={() => onBoxClick(box)}
          onHover={(hovered) => onBoxHover(hovered ? box : null)}
        />
      ))}

      <CogMarker metrics={metrics} containerL={L} containerW={W} />
      <gridHelper args={[Math.max(L, W) * 1.3, 40, '#94a3b8', '#cbd5e1']} />
    </>
  )
}

export function LoadingPlanViewer({ result, onNewRun }: LoadingPlanViewerProps) {
  const [selectedBoxId, setSelectedBoxId] = useState<string | null>(null)
  const [hoveredBox, setHoveredBox] = useState<PlacedBox | null>(null)

  const totalBoxes = result.placed_boxes.length
  const [currentStep, setCurrentStep] = useState<number>(totalBoxes || 1)
  const [isPlaying, setIsPlaying] = useState(false)
  const timerRef = useRef<number | null>(null)

  const metrics = result.metrics
  const container = result.container
  const L = container.internal_length_cm
  const W = container.internal_width_cm
  const H = container.internal_height_cm
  const camDist = Math.max(L, W, H)

  // Reset current step if result changes
  useEffect(() => {
    setCurrentStep(result.placed_boxes.length || 1)
    setIsPlaying(false)
    setSelectedBoxId(null)
  }, [result])

  // Sequence player animation loop
  useEffect(() => {
    if (isPlaying) {
      timerRef.current = window.setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalBoxes) {
            setIsPlaying(false)
            return totalBoxes
          }
          return prev + 1
        })
      }, 150)
    } else {
      if (timerRef.current) clearInterval(timerRef.current)
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
    }
  }, [isPlaying, totalBoxes])

  const displayedBoxes = useMemo(() => {
    return result.placed_boxes.filter((b) => (b.step_index ?? 1) <= currentStep)
  }, [result.placed_boxes, currentStep])

  const handleBoxClick = (box: PlacedBox) => {
    setSelectedBoxId(box.box_id === selectedBoxId ? null : box.box_id)
  }

  const selectedBox = useMemo(() => {
    return result.placed_boxes.find((b) => b.box_id === selectedBoxId)
  }, [result.placed_boxes, selectedBoxId])

  const activeInspectBox = hoveredBox || selectedBox

  const cameraPos: [number, number, number] = [L * 0.95, H * 1.8, W * 1.8]
  const controlsTarget: [number, number, number] = [0, H / 2, 0]

  return (
    <div className="viewer flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h2 className="text-xl font-bold text-slate-900">Loading Plan Result &ndash; {container.container_type}</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Dimensions: {L} &times; {W} &times; {H} cm &bull; Max Weight: {container.max_weight_kg.toLocaleString()} kg
          </p>
        </div>
        <button className="btn btn-primary" onClick={onNewRun}>
          Configure New Run
        </button>
      </div>

      {/* Applied Run Parameters (Advanced Options) */}
      <div className="flex items-center justify-between flex-wrap gap-2 px-4 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-700">
        <div className="flex items-center gap-2">
          <span className="flex items-center gap-1.5 font-semibold text-slate-800">
            <svg className="w-4 h-4 text-blue-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
            </svg>
            Applied Parameters:
          </span>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="inline-flex items-center gap-1 bg-white px-2.5 py-1 rounded-md border border-slate-200 font-medium shadow-sm">
              <span className="text-slate-400">Population:</span>
              <span className="text-blue-700 font-semibold">{result.options?.population_size ?? 30}</span>
            </span>
            <span className="inline-flex items-center gap-1 bg-white px-2.5 py-1 rounded-md border border-slate-200 font-medium shadow-sm">
              <span className="text-slate-400">Generations:</span>
              <span className="text-blue-700 font-semibold">{result.options?.generations ?? 40}</span>
            </span>
            <span className="inline-flex items-center gap-1 bg-white px-2.5 py-1 rounded-md border border-slate-200 font-medium shadow-sm">
              <span className="text-slate-400">Tolerance Gap:</span>
              <span className="text-blue-700 font-semibold">{result.options?.tolerance_gap_cm ?? 0.0} cm</span>
            </span>
          </div>
        </div>
        <span className="text-slate-400 text-[11px] font-mono">
          Run ID: {result.run_id}
        </span>
      </div>

      {/* Metrics Row (KPIs) */}
      {(() => {
        const containerVolCbm = (L * W * H) / 1_000_000
        const usedCbm = metrics.used_volume_cbm ?? (containerVolCbm * metrics.fill_rate)
        const unusedCbm = metrics.unused_volume_cbm ?? Math.max(0, containerVolCbm - usedCbm)
        const unusedPercent = Math.max(0, (1 - metrics.fill_rate) * 100).toFixed(1)
        
        let planningTime = '0.45'
        if (result.planning_time_seconds !== undefined && result.planning_time_seconds !== null) {
          planningTime = result.planning_time_seconds.toFixed(2)
        } else if (metrics.planning_time_seconds !== undefined && metrics.planning_time_seconds !== null) {
          planningTime = metrics.planning_time_seconds.toFixed(2)
        } else if (result.completed_at && result.created_at) {
          const diffSec = (new Date(result.completed_at).getTime() - new Date(result.created_at).getTime()) / 1000
          if (diffSec > 0) planningTime = diffSec.toFixed(2)
        }

        return (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="stat-card">
              <div className="stat-label">Utilization</div>
              <div className="stat-value text-blue-600">
                {(metrics.fill_rate * 100).toFixed(1)}<span className="stat-unit">%</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-medium">
                {usedCbm.toFixed(2)} / {containerVolCbm.toFixed(2)} CBM
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Unused CBM</div>
              <div className="stat-value text-slate-800">
                {unusedCbm.toFixed(2)}<span className="stat-unit"> CBM</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-medium">
                {unusedPercent}% remaining
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Planning Time</div>
              <div className="stat-value text-emerald-600">
                {planningTime}<span className="stat-unit">s</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-medium">
                Workflow run time
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Weight</div>
              <div className="stat-value text-slate-800">
                {(metrics.weight_utilization * 100).toFixed(1)}<span className="stat-unit">%</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-medium">
                {metrics.used_weight_kg.toLocaleString()} / {container.max_weight_kg.toLocaleString()} kg
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-label">Placed Cartons</div>
              <div className="stat-value text-slate-800">
                {displayedBoxes.length}<span className="stat-unit"> / {metrics.total_cartons}</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-medium">
                {Math.round((displayedBoxes.length / (metrics.total_cartons || 1)) * 100)}% placed
              </div>
            </div>

            <div className={`stat-card ${metrics.unplaced_count > 0 ? 'warning border-amber-300 bg-amber-50/30' : ''}`}>
              <div className="stat-label">Unplaced</div>
              <div className={`stat-value ${metrics.unplaced_count > 0 ? 'text-amber-600' : 'text-slate-800'}`}>
                {metrics.unplaced_count}<span className="stat-unit"> cartons</span>
              </div>
              <div className="text-[11px] text-slate-500 mt-1 font-medium">
                {metrics.unplaced_count > 0 ? 'Need extra space' : 'All loaded'}
              </div>
            </div>
          </div>
        )
      })()}

      {/* 3D Viewport Content */}
      <div className="viewer-content relative bg-slate-100 border border-slate-200 rounded-xl overflow-hidden shadow-sm" style={{ minHeight: '480px', height: '540px' }}>
        <Canvas
          camera={{ position: cameraPos, fov: 48, near: 1, far: camDist * 15 }}
          style={{ width: '100%', height: '100%' }}
          onCreated={({ gl }) => { gl.setClearColor('#f8fafc', 1) }}
        >
          <Scene
            placedBoxes={displayedBoxes}
            container={container}
            metrics={metrics}
            selectedBoxId={selectedBoxId}
            hoveredBoxId={hoveredBox?.box_id || null}
            currentStep={currentStep}
            onBoxClick={handleBoxClick}
            onBoxHover={setHoveredBox}
          />
          <OrbitControls
            target={controlsTarget}
            enableDamping
            dampingFactor={0.12}
            minDistance={30}
            maxDistance={camDist * 10}
          />
        </Canvas>

        {/* Floating Inspection HUD (Top Left) */}
        {activeInspectBox && (
          <div className="absolute top-4 left-4 z-10 bg-white/95 backdrop-blur border border-slate-200 text-slate-800 rounded-lg p-3 text-xs shadow-md max-w-xs pointer-events-none">
            <div className="flex items-center justify-between gap-2 border-b border-slate-100 pb-1.5 mb-1.5">
              <span className="font-semibold text-blue-600">Step #{activeInspectBox.step_index}</span>
              <span className="font-mono text-slate-400">{activeInspectBox.box_id}</span>
            </div>
            <div className="space-y-1">
              <div><span className="text-slate-400">SKU:</span> <span className="font-medium text-slate-800">{activeInspectBox.item_id}</span></div>
              {activeInspectBox.description && (
                <div><span className="text-slate-400">Desc:</span> <span className="text-slate-800">{activeInspectBox.description}</span></div>
              )}
              <div><span className="text-slate-400">PO:</span> <span className="text-slate-800">{activeInspectBox.po_no}</span></div>
              {activeInspectBox.customer_code && (
                <div><span className="text-slate-400">Customer:</span> <span className="text-emerald-600 font-medium">{activeInspectBox.customer_code}</span></div>
              )}
              <div><span className="text-slate-400">Dimensions:</span> <span className="text-slate-800">{activeInspectBox.actual_length} &times; {activeInspectBox.actual_width} &times; {activeInspectBox.actual_height} cm</span></div>
              <div><span className="text-slate-400">Weight:</span> <span className="text-slate-800">{activeInspectBox.weight_kg} kg</span></div>
              <div><span className="text-slate-400">Position (X,Y,Z):</span> <span className="font-mono text-slate-700">({activeInspectBox.x.toFixed(1)}, {activeInspectBox.y.toFixed(1)}, {activeInspectBox.z.toFixed(1)}) cm</span></div>
            </div>
          </div>
        )}
      </div>

      {/* Step-by-Step Sequence Slider Controller (Section 4) */}
      {totalBoxes > 0 && (
        <div className="card p-4 bg-white border border-slate-200 rounded-xl shadow-sm space-y-3">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm text-slate-800">Sequence Loading Sequence:</span>
              <span className="text-xs px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-700 font-medium">
                Step {currentStep} of {totalBoxes} ({Math.round((currentStep / totalBoxes) * 100)}%)
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={() => { setIsPlaying(false); setCurrentStep(1); }}
                title="Reset to step 1"
              >
                Reset
              </button>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={() => { setIsPlaying(false); setCurrentStep((prev) => Math.max(1, prev - 1)); }}
                disabled={currentStep <= 1}
                title="Step backward"
              >
                &larr; Prev
              </button>
              <button
                type="button"
                className={`btn btn-sm ${isPlaying ? 'btn-secondary' : 'btn-primary'}`}
                onClick={() => {
                  if (currentStep >= totalBoxes) setCurrentStep(1);
                  setIsPlaying(!isPlaying);
                }}
              >
                {isPlaying ? '⏸ Pause' : '▶ Play Sequence'}
              </button>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={() => { setIsPlaying(false); setCurrentStep((prev) => Math.min(totalBoxes, prev + 1)); }}
                disabled={currentStep >= totalBoxes}
                title="Step forward"
              >
                Next &rarr;
              </button>
              <button
                type="button"
                className="btn btn-outline btn-sm"
                onClick={() => { setIsPlaying(false); setCurrentStep(totalBoxes); }}
                title="Show full container"
              >
                Show All
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-400 font-mono">1</span>
            <input
              type="range"
              min={1}
              max={totalBoxes}
              value={currentStep}
              onChange={(e) => {
                setIsPlaying(false)
                setCurrentStep(parseInt(e.target.value, 10))
              }}
              className="flex-1 cursor-pointer accent-blue-600"
            />
            <span className="text-xs text-slate-400 font-mono">{totalBoxes}</span>
          </div>
        </div>
      )}

      {/* Selected Box Details Card */}
      {selectedBox && (
        <div className="p-4 bg-white border border-slate-200 rounded-xl shadow-sm text-sm">
          <div className="flex items-center justify-between border-b border-slate-100 pb-2 mb-3">
            <h3 className="font-semibold text-slate-800">Box Inspection &ndash; {selectedBox.box_id}</h3>
            <button className="text-xs text-slate-400 hover:text-slate-600" onClick={() => setSelectedBoxId(null)}>
              &times; Close
            </button>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
            <div><span className="text-slate-400">Item / SKU:</span> <div className="font-medium text-slate-800">{selectedBox.item_id}</div></div>
            <div><span className="text-slate-400">PO Number:</span> <div className="font-medium text-slate-800">{selectedBox.po_no}</div></div>
            <div><span className="text-slate-400">Customer:</span> <div className="font-medium text-slate-800">{selectedBox.customer_code || 'N/A'}</div></div>
            <div><span className="text-slate-400">Sequence Step:</span> <div className="font-semibold text-blue-600">#{selectedBox.step_index}</div></div>
            <div><span className="text-slate-400">Dimensions:</span> <div className="font-medium text-slate-800">{selectedBox.actual_length} &times; {selectedBox.actual_width} &times; {selectedBox.actual_height} cm</div></div>
            <div><span className="text-slate-400">Weight:</span> <div className="font-medium text-slate-800">{selectedBox.weight_kg} kg</div></div>
            <div><span className="text-slate-400">Logistics Position:</span> <div className="font-mono text-slate-800">({selectedBox.x.toFixed(1)}, {selectedBox.y.toFixed(1)}, {selectedBox.z.toFixed(1)})</div></div>
            <div><span className="text-slate-400">Orientation:</span> <div className="font-mono text-slate-800">{selectedBox.posture}</div></div>
          </div>
        </div>
      )}

      {/* Unplaced Cartons Section */}
      <UnplacedCartons unplacedCartons={result.unplaced_cartons} />
    </div>
  )
}