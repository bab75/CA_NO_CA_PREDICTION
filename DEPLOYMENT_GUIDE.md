# Deployment Guide for Chronic Absenteeism Prediction System

This guide will help you deploy the application to GitHub and Streamlit Cloud.

## GitHub Deployment

1. **Create a new GitHub repository**
   - Go to [GitHub](https://github.com) and log in to your account
   - Click "New repository" button
   - Name your repository (e.g., "chronic-absenteeism-prediction")
   - Choose visibility (public or private)
   - Click "Create repository"

2. **Initialize Git and push code**
   ```bash
   # Initialize Git repository (if not already done)
   git init
   
   # Add all files
   git add .
   
   # Commit changes
   git commit -m "Initial commit"
   
   # Add remote repository
   git remote add origin https://github.com/YOUR_USERNAME/chronic-absenteeism-prediction.git
   
   # Push to GitHub
   git push -u origin main
   ```

## Streamlit Cloud Deployment

1. **Create a Streamlit Cloud account**
   - Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign up/log in

2. **Connect your GitHub repository**
   - In Streamlit Cloud, click "New app"
   - Select your GitHub repository
   - In the deployment settings:
     - Main file path: `app.py`
     - Python version: 3.11
   - Click "Deploy"

3. **Access your deployed application**
   - Streamlit Cloud will provide a URL where your application is deployed
   - Share this URL with others to access your application

## Important Notes for Deployment

1. Ensure your `requirements.txt` file (included in the zip) contains all necessary dependencies.
2. The `.streamlit/config.toml` file is included for proper server configuration.
3. You can customize the README.md file to include more details about your project.
4. Your application is structured to work with both GitHub and Streamlit Cloud.

## Migrating from Replit to GitHub

1. Use the provided `chronic_absenteeism_prediction_app.zip` file.
2. Extract the contents to your local machine.
3. Follow the GitHub deployment steps above.