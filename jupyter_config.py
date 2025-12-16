"""Jupyter configuration for AI Fine-tuning Project."""

# Configuration file for jupyter-lab
c = get_config()  # noqa

# Set the default directory for notebooks
c.ServerApp.root_dir = '.'

# Enable extensions
c.ServerApp.jpserver_extensions = {
    'jupyterlab': True
}

# Security settings
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.open_browser = True

# Port configuration
c.ServerApp.port = 8888
c.ServerApp.port_retries = 50

# Allow remote connections (set to False for security in production)
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_remote_access = True

# Notebook settings
c.FileContentsManager.delete_to_trash = False

# Terminal settings
c.ServerApp.terminals_enabled = True

# File browser settings
c.ContentsManager.allow_hidden = False
