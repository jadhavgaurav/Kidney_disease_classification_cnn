# 🚀 Deploy Flask + ML App on AWS EC2 using Gunicorn + Nginx

Guide for deploying a Flask-based machine learning application on an AWS EC2 instance using **Gunicorn** and **Nginx** in a production environment.

---

## 🔧 Prerequisites

- AWS account with access to EC2
- SSH key (`.pem`) to connect to your EC2 instance
- Flask ML app (with `app.py`, `templates/`, `static/`, `requirements.txt`)
- Trained ML model (e.g., `.h5`, `joblib`, or MLflow)
- GitHub repository (optional)

---

## 1️⃣ Launch EC2 Instance

- OS: Ubuntu 22.04 LTS
- Instance type: `t2.micro` (Free Tier)
- Add Security Group Rules:
  - `HTTP (80)`
  - `Custom TCP` (Ports: 8000, 8080, 5000 if needed)
  - `SSH (22)` from your IP

---

## 2️⃣ SSH into Instance

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<your-ec2-public-ip>
```

---

## 3️⃣ Install System Packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip python3-venv nginx git -y
```

---

## 4️⃣ Clone Your Flask Project

```bash
git clone https://github.com/yourusername/your-repo.git
cd your-repo
```

---

## 5️⃣ Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6️⃣ Test Flask App (Optional)

```bash
python3 app.py
```

Access: `http://<your-ec2-public-ip>:5000`

---

## 7️⃣ Create WSGI Entry Point

Create `wsgi.py`:

```python
from app import app

if __name__ == "__main__":
    app.run()
```

---

## 8️⃣ Run Gunicorn (App Server)

```bash
gunicorn --bind 0.0.0.0:8000 wsgi:app
```

Test at: `http://<ec2-public-ip>:8000`

---

## 9️⃣ Create a Gunicorn Systemd Service

```bash
sudo nano /etc/systemd/system/kidney_classifier.service
```

Paste below content:

```ini
[Unit]
Description=Gunicorn instance to serve Flask app
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/your-repo
Environment="PATH=/home/ubuntu/your-repo/venv/bin"
ExecStart=/home/ubuntu/your-repo/venv/bin/gunicorn --workers 3 --bind 127.0.0.1:8000 wsgi:app

[Install]
WantedBy=multi-user.target
```

Start & Enable:

```bash
sudo systemctl daemon-reload
sudo systemctl enable kidney_classifier
sudo systemctl start kidney_classifier
```

Check status:

```bash
sudo systemctl status kidney_classifier
```

---

## 🔟 Set Up Nginx as a Reverse Proxy

Create config:

```bash
sudo nano /etc/nginx/sites-available/kidney_classifier
```

Paste config:

```nginx
server {
    listen 80;
    server_name your-ec2-public-ip;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Enable site:

```bash
sudo ln -s /etc/nginx/sites-available/kidney_classifier /etc/nginx/sites-enabled
sudo nginx -t
sudo systemctl restart nginx
```

Visit your app:
```
http://<your-ec2-public-ip>
```

---

## ✅ Optional Final Touches

- Secure with SSL using Let's Encrypt + Certbot
- Add domain name in `server_name`
- Use `.env` or AWS Secrets for environment variables

---

## 🧼 Cleanup (After Testing)

- Revert Security Group to allow only `HTTP (80)` & `SSH (22)`
- Monitor app logs (`journalctl`, `gunicorn`, `nginx`, etc.)

---

## 📌 Notes

- `Gunicorn` handles production WSGI server functionality
- `Nginx` acts as a reverse proxy and handles incoming traffic
- Avoid using `python3 app.py` or `screen` in production

---

**Built with ❤️ by Gaurav Jadhav**
