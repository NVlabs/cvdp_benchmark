FROM rockylinux/rockylinux:8.10

WORKDIR /

# Add your Cadence tool archives here
# These should be obtained from Cadence support portal.
# The tools can be installed locally and then tgz'ed and added to the dockerfile.
# Replace with your actual tool archive names and versions
ADD vmanager.tgz .
ADD xcelium.tgz .

RUN dnf update -y && dnf upgrade -y
RUN dnf install -y ksh tcsh redhat-lsb-core redhat-lsb-core.i686

# Need for Vmanager
RUN dnf install -y libnsl.x86_64

# Include Xcelium in PATH
# Update these paths based on your tool versions and installation locations
# Format: /path/to/cadence/XCELIUM<VERSION>/bin
ENV PATH="$PATH:/path/to/cadence/XCELIUM<VERSION>/bin"
ENV PATH="$PATH:/path/to/cadence/XCELIUM<VERSION>/tools.lnx86/bin/64bit"

# Include Vmanager in PATH  
# Update this path based on your Vmanager version
# Format: /path/to/cadence/VMANAGER<VERSION>/bin
ENV PATH="$PATH:/path/to/cadence/VMANAGER<VERSION>/bin"

# Configuration of the license
# IMPORTANT: Replace with your organization's license server(s)
# Format: port@server1:port@server2:port@server3
# Example: ENV CDS_LIC_FILE=5280@license-server-1:5280@license-server-2
ENV CDS_LIC_FILE=REPLACE_WITH_YOUR_LICENSE_SERVERS
ENV LM_LICENSE_FILE="$CDS_LIC_FILE"

# ----------------------------------------
# - Preparing Python Environment
# ----------------------------------------

RUN dnf -y install python3.12
ADD https://bootstrap.pypa.io/get-pip.py get-pip.py
RUN python3 ./get-pip.py
RUN python3 -m pip install --no-cache-dir \
    pytest==9.0.2 \
    pytest-timeout==2.4.0 \
    cocotb==2.0.1 \
    coverage \
    aes
