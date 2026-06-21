module.exports = {
  apps: [
    {
      name: "livekit-fastapi-api",
      cwd: "/root/Livekit-voice-ai", 
      script: "/root/Livekit-voice-ai/.venv/bin/uvicorn",
      args: "main:app --host 0.0.0.0 --port 8888",
      interpreter: "none",         // <--- This is the magic line
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
