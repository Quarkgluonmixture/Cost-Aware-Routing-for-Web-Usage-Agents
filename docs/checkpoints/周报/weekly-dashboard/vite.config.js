import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  base: './',  // 相对路径，dist/index.html 可直接打开
  plugins: [tailwindcss(), react()],
})
