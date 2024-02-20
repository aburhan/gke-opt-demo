import os
import requests

def create_github_issue(repo_owner, repo_name, issue_title, issue_body, token):
    """
    Create an issue on github.com using the given parameters.
    """
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {
        "title": issue_title,
        "body": issue_body
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 201:
        print(f"Issue created successfully: {response.json()['html_url']}")
    else:
        print(f"Failed to create issue: {response.content}")

if __name__ == "__main__":
    first_name = os.environ.get('FIRST_NAME')
    last_name = os.environ.get('LAST_NAME')
    token = os.environ.get('GITHUB_TOKEN')  # GitHub token for authentication

    # Ensure you replace 'repo_owner' and 'repo_name' with your GitHub username and repository name
    repo_owner = 'your_username'
    repo_name = 'your_repo_name'
    
    issue_title = "New Issue Created via GitHub API"
    issue_body = f"Hello, this issue was created automatically. Name: {first_name} {last_name}"

    create_github_issue(repo_owner, repo_name, issue_title, issue_body, token)
