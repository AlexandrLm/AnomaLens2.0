import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000, // Укажем порт для фронтенда (например, 3000)
    proxy: {
      // Строка '/api' будет проксироваться: '/api/...' -> 'http://127.0.0.1:8001/api/...'
      '/api': {
        target: 'http://127.0.0.1:8001', // Наш бэкенд
        changeOrigin: true, // Необходимо для виртуальных хостов
        secure: false,      // Не проверять SSL сертификат (если бэкенд на HTTPS)
        // rewrite: (path) => path.replace(/^\/api/, ''), // Если нужно убрать /api из пути к бэкенду
      }
    }
  }
})
