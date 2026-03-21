# Docker Deployment Plan - Backend Only (Production Ready)

**Date**: March 20, 2026  
**Project**: RAD-ML-v8  
**Scope**: Deploy production-ready backend API to Docker (localhost)  
**Frontend**: NOT included (optional for future)

---

## 📋 Overview

This document outlines the complete implementation plan to containerize **ONLY the production-ready backend API** for deployment with Docker. The backend includes:

- ✅ **Flask REST API** - Authentication, Pipeline, Chat History
- ✅ **Code Generator (RAD-ML)** - Model generation engine
- ✅ **Data Collection Agent** - Kaggle + OpenML + HuggingFace
- ❌ **Frontend** - Not included (can run separately)
- ❌ **Tests** - Not included for production

---

## 🎯 Deployment Strategy

### What Gets Containerized
```
Container (Docker)
├── Flask Backend (Port 5000)
│   ├── Auth Module
│   ├── Pipeline Endpoints
│   └── Chat History
├── Code Generator (RAD-ML)
├── Data Collection Agent
│   ├── Kaggle Collector
│   ├── OpenML Collector
│   ├── UCI Collector
│   └── HuggingFace Collector
└── Volumes (Persisted)
    ├── /data (collected datasets)
    └── /logs (application logs)
```

### What Stays Local (Not in Container)
- Frontend (React/Vite) - Optional
- Tests & pytest files - For development only
- Documentation files
- IDE configurations

---

## 🔧 Step 1: Create Dockerfile.backend

**Location**: `<project-root>/Dockerfile.backend`

```dockerfile
# Multi-stage build to keep image lean
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary backend/model files
COPY ./requirements.txt .
COPY ./config.yaml .
COPY ./Chatbot_Interface/backend/ ./Chatbot_Interface/backend/
COPY ./Code_Generator/ ./Code_Generator/
COPY ./Data_Collection_Agent/ ./Data_Collection_Agent/
COPY ./API_KEYS.md ./API_KEYS.md

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health')" || exit 1

# Set environment variables
ENV FLASK_APP=Chatbot_Interface.backend.app
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Run backend
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
```

**Key Points**:
- Uses Python 3.11-slim (lightweight base image)
- Only copies production code (excludes tests, frontend, etc.)
- Health check enabled every 30 seconds
- Flask runs on 0.0.0.0:5000 (accessible from host)
- PYTHONUNBUFFERED ensures logs appear immediately

---

## 🔧 Step 2: Create docker-compose.yml

**Location**: `<project-root>/docker-compose.yml`

```yaml
version: '3.9'

services:
  rad-ml-backend:
    # Build from Dockerfile
    build:
      context: .
      dockerfile: Dockerfile.backend
    
    # Container name
    container_name: rad-ml-api
    
    # Port mapping: container:host
    ports:
      - "5000:5000"           # Expose backend on localhost:5000
    
    # Environment variables
    environment:
      - FLASK_ENV=production
      - PYTHONUNBUFFERED=1
      - JWT_SECRET_KEY=${JWT_SECRET_KEY:-your-secret-key-here}
    
    # Mount persistent volumes
    volumes:
      # Collect data from datasets
      - ./Chatbot_Interface/backend/data:/app/Chatbot_Interface/backend/data
      # Persist application logs
      - ./logs:/app/logs
    
    # Load environment variables from .env file
    env_file:
      - .env
    
    # Restart policy
    restart: unless-stopped
    
    # Assign to network
    networks:
      - rad-ml-network

# Network definition
networks:
  rad-ml-network:
    driver: bridge
```

**Features**:
- Single service (backend only)
- Automatic restart if container crashes
- Volumes persist data even when container stops
- Environment variables loaded from `.env` file
- Custom network for isolation

---

## 🔧 Step 3: Create .dockerignore

**Location**: `<project-root>/.dockerignore`

Prevents unnecessary files from being copied into the Docker image:

```
# Version control
.git
.gitignore

# Environment files
.env.local
.env.*.local

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
*.egg-info
dist
build
.pytest_cache
.venv
venv

# Node.js (frontend - not needed)
node_modules
Chatbot_Interface/frontend

# Testing
tests
*.test.js
pytest.ini
conftest.py

# Documentation
*.md
docs/

# OS
.DS_Store
Thumbs.db

# Logs and caches
*.log
.coverage
.mypy_cache
```

**Why This Matters**:
- Reduces image size significantly
- Faster build times
- Excludes unnecessary files
- Frontend completely excluded

---

## 🔧 Step 4: Create Environment Configuration

### 4a. Create `.env.example`

**Location**: `<project-root>/.env.example`

Template for environment variables (commit this to git):

```env
# ===========================
# Flask Configuration
# ===========================
FLASK_ENV=production
FLASK_DEBUG=0

# JWT Secret Key (CHANGE THIS IN PRODUCTION!)
JWT_SECRET_KEY=your-production-secret-key-change-this-to-random-string

# ===========================
# Database Configuration
# ===========================
# MongoDB (for job history persistence)
MONGODB_URI=mongodb://localhost:27017/rad-ml

# ===========================
# Data Collection Credentials
# ===========================
# Kaggle API
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_key

# ===========================
# LLM Configuration
# ===========================
# Google Generative AI
GOOGLE_API_KEY=your_google_generative_ai_key

# ===========================
# AWS Configuration (Optional - for SageMaker)
# ===========================
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1

# ===========================
# Application Configuration
# ===========================
LOG_LEVEL=INFO
MAX_JOBS_PER_USER=50
```

### 4b. Create `.env` (DO NOT COMMIT)

**Location**: `<project-root>/.env`

Actual environment variables with real values:

```bash
# Copy from .env.example and fill in your actual values
FLASK_ENV=production
FLASK_DEBUG=0

# Generate a random secret key using:
# python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0

MONGODB_URI=mongodb://localhost:27017/rad-ml

KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key_here

GOOGLE_API_KEY=your_api_key_here

AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_DEFAULT_REGION=us-east-1

LOG_LEVEL=INFO
MAX_JOBS_PER_USER=50
```

**Security**: Add `.env` to `.gitignore`:

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore
```

---

## 🔧 Step 5: Deployment Commands

### 5a. Build the Docker Image

```powershell
# Navigate to project root
cd "c:\Users\sabhi\OneDrive\Desktop\RAD-ML-v8"

# Build the image
docker-compose build

# Output should show:
# [+] Building 120.5s (15/15) FINISHED
# => rad-ml-backend
```

**What Happens**:
1. Reads Dockerfile.backend
2. Installs Python 3.11-slim base image
3. Copies project files (excluding .dockerignore)
4. Installs Python dependencies from requirements.txt
5. Creates image tagged as `rad-ml-backend:latest`

### 5b. Start the Container

```powershell
# Start in detached mode (runs in background)
docker-compose up -d

# Output:
# [+] Running 1/1
# ✓ rad-ml-api Started
```

### 5c. Verify Deployment

```powershell
# Check running containers
docker-compose ps

# Output:
# NAME        COMMAND                 SERVICE    STATUS
# rad-ml-api  python -m flask run...  backend    Up 5 seconds
```

### 5d. View Logs

```powershell
# Follow logs in real-time
docker-compose logs -f rad-ml-backend

# View last 100 lines
docker-compose logs --tail=100 rad-ml-backend

# Sample output:
# rad-ml-api  | WARNING: This is a development server. Do not use it in production.
# rad-ml-api  | Running on http://0.0.0.0:5000
# rad-ml-api  | Press CTRL+C to quit
```

### 5e. Stop the Container

```powershell
# Stop running container
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Output:
# [+] Running 1/0
# ✓ rad-ml-api Stopped
```

---

## 🔧 Step 6: API Testing & Verification

### 6a. Health Check

```powershell
# Test if API is healthy
Invoke-WebRequest http://localhost:5000/api/health -Method GET

# Expected response:
# StatusCode        : 200
# Content           : {"status":"ok"}
```

### 6b. Register New User

```powershell
# Create user
$body = @{
    username = "testuser"
    password = "testpass123"
    email = "test@example.com"
} | ConvertTo-Json

$response = Invoke-WebRequest -Uri http://localhost:5000/api/auth/register `
  -Method POST `
  -ContentType "application/json" `
  -Body $body

$response.Content | ConvertFrom-Json

# Expected response:
# {
#   "msg": "User registered successfully"
# }
```

### 6c. Login

```powershell
# Login with credentials
$loginBody = @{
    username = "testuser"
    password = "testpass123"
} | ConvertTo-Json

$loginResponse = Invoke-WebRequest -Uri http://localhost:5000/api/auth/login `
  -Method POST `
  -ContentType "application/json" `
  -Body $loginBody

# Parse response and extract token
$responseData = $loginResponse.Content | ConvertFrom-Json
$token = $responseData.access_token

Write-Host "JWT Token: $token"

# Expected response:
# {
#   "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
#   "username": "testuser"
# }
```

### 6d. Run Pipeline

```powershell
# Define the prompt
$pipelineBody = @{
    prompt = "Create a machine learning model that predicts house prices based on features like size, location, and age"
} | ConvertTo-Json

# Call pipeline endpoint with JWT token
$token = "your-jwt-token-from-login"

$pipelineResponse = Invoke-WebRequest -Uri http://localhost:5000/api/pipeline/run `
  -Method POST `
  -ContentType "application/json" `
  -Headers @{ Authorization = "Bearer $token" } `
  -Body $pipelineBody

$pipelineResponse.Content | ConvertFrom-Json

# Expected response:
# {
#   "job_id": "job_12345",
#   "status": "processing"
# }
```

### 6e. Check Pipeline Status

```powershell
# Check job status
$token = "your-jwt-token"
$jobId = "job_12345"

$statusResponse = Invoke-WebRequest -Uri "http://localhost:5000/api/pipeline/status/$jobId" `
  -Method GET `
  -Headers @{ Authorization = "Bearer $token" }

$statusResponse.Content | ConvertFrom-Json

# Expected response:
# {
#   "job_id": "job_12345",
#   "status": "completed",
#   "accuracy": 0.9797,
#   "model_path": "/app/models/job_12345.pkl"
# }
```

---

## 📊 Architecture Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                         Docker Container                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Flask REST API Server (Port 5000)               │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │                                                          │ │
│  │  ┌──────────────────┐  ┌──────────────────────────────┐ │ │
│  │  │  Auth Module     │  │  Pipeline Controller         │ │ │
│  │  ├──────────────────┤  ├──────────────────────────────┤ │ │
│  │  │ • Register       │  │ • /pipeline/run              │ │ │
│  │  │ • Login          │  │ • /pipeline/status/<id>      │ │ │
│  │  │ • JWT Tokens     │  │ • /pipeline/stream/<id> (SSE)│ │ │
│  │  │ • Get User       │  │                              │ │ │
│  │  └──────────────────┘  └──────────────────────────────┘ │ │
│  │                                                          │ │
│  │  ┌──────────────────────────────────────────────────┐   │ │
│  │  │  Chat History Module                             │   │ │
│  │  ├──────────────────────────────────────────────────┤   │ │
│  │  │ • GET /api/history (list user jobs)              │   │ │
│  │  │ • GET /api/history/<id> (job details)            │   │ │
│  │  │ • DELETE /api/history/<id>                       │   │ │
│  │  └──────────────────────────────────────────────────┘   │ │
│  │                                                          │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Code Generator (RAD-ML)                         │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ • Prompt Understanding                                  │ │
│  │ • Code Planning                                         │ │
│  │ • Code Generation                                       │ │
│  │ • Code Verification                                     │ │
│  │ • Model Training                                        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │      Data Collection Agent (4-Tier Fallback)            │ │
│  ├──────────────────────────────────────────────────────────┤ │
│  │ Tier 1: Kaggle + UCI + OpenML (Parallel)                │ │
│  │         ↓                                                │ │
│  │ Tier 2: Kaggle Fallback References                      │ │
│  │         ↓                                                │ │
│  │ Tier 3: OpenML Extended Search                          │ │
│  │         ↓                                                │ │
│  │ Tier 4: HuggingFace Hub (99.9% Uptime Guaranteed)      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
└────────────────────────────────────────────────────────────────┘
                        ↕ (Port 5000)
┌────────────────────────────────────────────────────────────────┐
│                  Host Machine (Windows)                        │
├────────────────────────────────────────────────────────────────┤
│  localhost:5000  ← API accessible from any client             │
│                                                                │
│  MOUNTED VOLUMES:                                              │
│  • ./Chatbot_Interface/backend/data ← Collected datasets      │
│  • ./logs ← Application logs                                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 📁 Final Project Structure

After implementation:

```
RAD-ML-v8/
│
├── Dockerfile.backend              ✨ NEW
├── docker-compose.yml              ✨ NEW
├── .dockerignore                   ✨ NEW
├── .env                            ✨ NEW (DO NOT COMMIT)
├── .env.example                    ✨ NEW (COMMIT THIS)
│
├── requirements.txt                (existing)
├── config.yaml                     (existing)
├── API_KEYS.md                     (existing)
│
├── Chatbot_Interface/
│   ├── backend/                    ← Containerized
│   │   ├── app.py
│   │   ├── auth_db.py
│   │   ├── chat_history_db.py
│   │   ├── orchestrator.py
│   │   └── data/
│   │
│   └── frontend/                   (NOT containerized)
│
├── Code_Generator/                 ← Containerized
│   └── RAD-ML/
│
├── Data_Collection_Agent/          ← Containerized
│
├── tests/                          (NOT containerized)
├── logs/                           (mounted volume)
│
└── DOCKER_DEPLOYMENT_PLAN.md       ✨ NEW (this file)
```

---

## ✅ Quick Start Checklist

Follow these steps in order:

- [ ] **Step 1**: Create `Dockerfile.backend` from this plan
- [ ] **Step 2**: Create `docker-compose.yml` from this plan
- [ ] **Step 3**: Create `.dockerignore` from this plan
- [ ] **Step 4a**: Create `.env.example` with template values
- [ ] **Step 4b**: Create `.env` with your actual API keys
- [ ] **Step 5a**: Run `docker-compose build`
- [ ] **Step 5b**: Run `docker-compose up -d`
- [ ] **Step 5c**: Run `docker-compose ps` (verify running)
- [ ] **Step 5d**: Run `docker-compose logs -f` (check for errors)
- [ ] **Step 6a**: Test `http://localhost:5000/api/health`
- [ ] **Step 6b**: Test register endpoint
- [ ] **Step 6c**: Test login endpoint
- [ ] **Step 6d**: Test pipeline endpoint
- [ ] **Step 6e**: Verify model accuracy in response

---

## 🔐 Security Best Practices

### JWT Secret Key Generation

Generate a cryptographically secure secret:

```powershell
# Using Python
python -c "import secrets; print(secrets.token_hex(32))"

# Expected output:
# a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
```

Add to `.env`:
```env
JWT_SECRET_KEY=a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6
```

### .env File Protection

```bash
# Add to .gitignore (DO THIS FIRST!)
echo ".env" >> .gitignore
echo ".env.local" >> .gitignore

# Verify
cat .gitignore | grep ".env"

# Should output:
# .env
# .env.local
```

### API Key Security

**NEVER**:
- ❌ Hardcode API keys in code
- ❌ Commit `.env` file to git
- ❌ Use same secret in dev and production
- ❌ Log sensitive information
- ❌ Expose secrets in Docker logs

**ALWAYS**:
- ✅ Use environment variables
- ✅ Rotate secrets periodically
- ✅ Use `.env.example` as template
- ✅ Restrict container logs
- ✅ Use secrets management (AWS Secrets Manager, Azure Key Vault, etc.)

---

## 🐛 Troubleshooting

### Problem: "Port 5000 already in use"

```powershell
# Find process using port 5000
Get-NetTCPConnection -LocalPort 5000

# Kill the process (PowerShell as Admin)
Stop-Process -Id <PID> -Force

# Or use docker-compose with different port
# Edit docker-compose.yml:
# ports:
#   - "5001:5000"  ← Change to 5001
```

### Problem: "ModuleNotFoundError: No module named 'kaggle'"

```powershell
# Check if dependencies installed
docker-compose exec rad-ml-backend pip list | findstr kaggle

# If missing, rebuild
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Problem: "Docker daemon not running"

```powershell
# Start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Wait 30 seconds for startup
Start-Sleep -Seconds 30

# Verify
docker ps
```

### Problem: "Connection refused on localhost:5000"

```powershell
# Check if container is running
docker-compose ps

# If not running
docker-compose up -d

# If running but not responding
docker-compose logs rad-ml-backend

# Check published ports
docker-compose port rad-ml-backend 5000
```

### Problem: "Kaggle authentication failed"

```powershell
# Verify .env has correct credentials
cat .env | findstr KAGGLE

# Check if kaggle.json exists
dir "$env:USERPROFILE\.kaggle\kaggle.json"

# If missing, download from kaggle.com/settings/account
# Then copy to ~/.kaggle/kaggle.json
```

---

## 📊 Performance Monitoring

### Monitor Container Resource Usage

```powershell
# Real-time stats
docker stats

# Specific container
docker stats rad-ml-api

# Output:
# CONTAINER    CPU %    MEM USAGE   MEM %    NET I/O
# rad-ml-api   2.5%     450MiB       45%    2.1MB/500MB
```

### Check Container Logs for Issues

```powershell
# Last 50 lines
docker-compose logs --tail=50

# Follow in real-time
docker-compose logs -f

# Since specific time
docker-compose logs --since 10m
```

### Monitor API Response Times

```powershell
# Using curl (if installed)
curl -i -w "\nTime: %{time_total}s\n" http://localhost:5000/api/health

# Using PowerShell Measure-Object
Measure-Command {
    Invoke-WebRequest http://localhost:5000/api/health
}
```

---

## 🚀 Production Deployment Steps

When ready to deploy to production:

1. **Update .env with production values**
   ```env
   FLASK_ENV=production
   JWT_SECRET_KEY=<production-random-string>
   LOG_LEVEL=WARNING
   ```

2. **Use production-grade database**
   ```yaml
   # docker-compose.yml
   services:
     mongodb:
       image: mongo:7.0
       ports:
         - "27017:27017"
       volumes:
         - mongo_data:/data/db
   ```

3. **Add reverse proxy (Nginx)**
   ```yaml
   # docker-compose.yml
   services:
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
   ```

4. **Enable HTTPS**
   ```yaml
   # Use Let's Encrypt + Certbot
   # Or cloud provider SSL certificate
   ```

5. **Set resource limits**
   ```yaml
   # docker-compose.yml
   services:
     rad-ml-backend:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 2G
           reservations:
             cpus: '1'
             memory: 1G
   ```

---

## 📞 Support & Help

### View Full Container Logs

```powershell
docker-compose logs rad-ml-backend --all
```

### Execute Command Inside Container

```powershell
# List files
docker-compose exec rad-ml-backend ls -la /app

# Check Python version
docker-compose exec rad-ml-backend python --version

# Install additional package
docker-compose exec rad-ml-backend pip install <package>
```

### SSH into Container

```powershell
# Start bash shell
docker-compose exec rad-ml-backend /bin/bash

# Now inside container:
# ls -la
# pip list
# exit
```

---

## ✅ Implementation Summary

| Step | Task | File | Status |
|------|------|------|--------|
| 1 | Create Dockerfile | `Dockerfile.backend` | Create |
| 2 | Create docker-compose | `docker-compose.yml` | Create |
| 3 | Create .dockerignore | `.dockerignore` | Create |
| 4a | Create env template | `.env.example` | Create |
| 4b | Create env secrets | `.env` | Create (DO NOT COMMIT) |
| 5 | Build & Deploy | Run `docker-compose` | Execute |
| 6 | Test endpoints | API calls | Verify |

---

## 🎉 Next Steps

1. **Prepare files** (Steps 1-4)
2. **Build container** (Step 5a-5b)
3. **Test locally** (Step 6)
4. **Monitor logs** (Step 5d)
5. **Deploy to production** (when ready)

---

**Last Updated**: March 20, 2026  
**Status**: Ready for Implementation ✅
