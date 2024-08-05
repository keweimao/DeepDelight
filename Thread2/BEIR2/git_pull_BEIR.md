# Step-by-Step Local Deployment of BEIR Code

Mengyang Xu\
8/5/2024

**Basically shows how to pull the elastic code structure from the deepdelight github.\

Initializes a new Git repository in the current directory.
```cmd
git init
```

Adds a new remote repository and fetches all its branches.
```cmd
git remote add -f origin https://github.com/keweimao/DeepDelight.git  
```

Initializes sparse-checkout with a "cone" mode, which simplifies specifying directories to be included.
```cmd
git sparse-checkout init --cone
```

Defines which directories should be included in the sparse-checkout.
```cmd
git sparse-checkout set DeepDelight/Thread2/BEIR
```
Fetches and integrates changes from the remote repository's main branch into the local repository.
```cmd
git pull origin main
```
