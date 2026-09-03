/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Unset in development, which is what switches the app onto fixtures. */
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
