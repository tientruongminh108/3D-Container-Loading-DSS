import { Component, ErrorInfo, ReactNode } from 'react'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
  }

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Uncaught error caught by ErrorBoundary:', error, errorInfo)
  }

  public handleReset = () => {
    this.setState({ hasError: false, error: null })
    window.location.reload()
  }

  public render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback
      }
      return (
        <div className="p-8 max-w-xl mx-auto my-12 bg-white border border-red-200 rounded-xl shadow-sm text-center">
          <div className="w-12 h-12 rounded-full bg-red-100 text-red-600 flex items-center justify-center mx-auto mb-4">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="12" />
              <line x1="12" y1="16" x2="12.01" y2="16" />
            </svg>
          </div>
          <h2 className="text-lg font-bold text-slate-900 mb-2">Something went wrong</h2>
          <p className="text-sm text-slate-600 mb-4">
            An unexpected error occurred while rendering this view.
          </p>
          {this.state.error?.message && (
            <div className="p-3 mb-6 bg-slate-50 border border-slate-200 rounded text-xs font-mono text-left text-red-600 overflow-x-auto">
              {this.state.error.message}
            </div>
          )}
          <div className="flex justify-center gap-3">
            <button
              onClick={this.handleReset}
              className="btn btn-primary btn-sm"
            >
              Reload Page
            </button>
            <a href="/dashboard" className="btn btn-outline btn-sm">
              Go to Dashboard
            </a>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
