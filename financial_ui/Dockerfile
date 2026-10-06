FROM nginx:alpine

RUN sed -i 's/worker_processes[[:space:]]*auto;/worker_processes 1;/' /etc/nginx/nginx.conf

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html styles.css workspace.css app.js workspace.js notebooks.js settings.html settings.css settings.js icon.png favicon.svg refresh-icon-545661.png /usr/share/nginx/html/

EXPOSE 80
