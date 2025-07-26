# Script pour mettre à jour le fichier config.json avec l'URL de l'API
# Script dans dossier app_streamlit :

# python3 update_config.py


import subprocess
import json

# Paramètres à adapter si besoin :
CLUSTER_NAME = "projet5stackoverflow-cluster"
SERVICE_NAME = "projet5stackoverflow-flask-service"
CONFIG_JSON = "config.json"
API_PORT = 5001

def run_cmd(cmd):
    result = subprocess.check_output(cmd, shell=True)
    return result.decode('utf-8').strip()

# 1. Trouver la task "RUNNING" du service ECS
task_arn = run_cmd(
    f"aws ecs list-tasks --cluster {CLUSTER_NAME} --service-name {SERVICE_NAME} --desired-status RUNNING --output text --query 'taskArns[0]'"
)
if not task_arn:
    raise Exception("Aucune task RUNNING trouvée.")

# 2. Récupérer l'ENI (Elastic Network Interface) de la task
eni_id = run_cmd(
    f"aws ecs describe-tasks --cluster {CLUSTER_NAME} --tasks {task_arn} "
    f"--query 'tasks[0].attachments[0].details[?name==`networkInterfaceId`].value' --output text"
)

# 3. Récupérer l'IP publique de l'ENI
ip_publique = run_cmd(
    f"aws ec2 describe-network-interfaces --network-interface-ids {eni_id} "
    f"--query 'NetworkInterfaces[0].Association.PublicIp' --output text"
)

print(f"Nouvelle IP publique : {ip_publique}")

# 4. Met à jour config.json
url_api = f"http://{ip_publique}:{API_PORT}/predict"
with open(CONFIG_JSON, "w") as f:
    json.dump({"URL_API": url_api}, f, indent=4, ensure_ascii=False)

print(f"{CONFIG_JSON} mis à jour avec : {url_api}")

# 5. Commit et push Git
subprocess.run(["git", "add", CONFIG_JSON])
subprocess.run(["git", "commit", "-m", f"Update API URL to {ip_publique}"])
subprocess.run(["git", "push"])

print("Commit & push faits. C'est prêt !")
