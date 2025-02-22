#!/bin/bash

# installing nvidia-container-toolkit
nvidia_toolkit_installation() {
	echo "## Preparing environment for Docker container be able to use the GPU...";
	echo "# Installing nvidia-container-toolkit";

	curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
	  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
	    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
	    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list;

	# now, installing...
	sudo apt-get update -y
	sudo apt-get install -y nvidia-container-toolkit

	# configuring docker runtime
	sudo nvidia-ctk runtime configure --runtime=docker

	# restart docker daemon
	sudo systemctl restart docker

	echo "Nvidia-Container-Toolkit successfuly installed"
}

# creating and defining contents for our Dockerfile
create_dockerfile() {
	echo "## Creating Dockerfile..."

	touch Dockerfile
	echo "FROM debian" >> Dockerfile
	echo "WORKDIR /comfyUI-server" >> Dockerfile
	echo \ >> Dockerfile
	echo "# INSTALLATIONS" >> Dockerfile
	echo "RUN apt update -y && apt install -y \\" >> Dockerfile
	echo "	python3-venv \\" >> Dockerfile
	echo "	git \\" >> Dockerfile
	echo "	python3-pip \\" >> Dockerfile
	echo "	wget \\" >> Dockerfile
	echo "	unzip \\" >> Dockerfile
	echo "	libgl1-mesa-glx \\" >> Dockerfile
	echo "	libglib2.0-0" >> Dockerfile
	echo \ >> Dockerfile
	echo "RUN useradd -m -s /bin/bash comfy-user && chown comfy-user /comfyUI-server" >> Dockerfile
	echo \ >> Dockerfile
	echo "USER comfy-user" >> Dockerfile
	echo \ >> Dockerfile
	echo "# MODELS DOWNLOADER" >> Dockerfile
	echo "RUN wget https://github.com/Acly/krita-ai-diffusion/releases/download/v1.31.1/krita_ai_diffusion-1.31.1.zip && unzip krita_ai_diffusion-1.31.1.zip -d krita_ai_diffusion-1.31.1" >> Dockerfile
	echo \ >> Dockerfile
	echo "RUN git clone --depth 1 --single-branch https://github.com/comfyanonymous/ComfyUI.git" >> Dockerfile
	echo \ >> Dockerfile
	echo "RUN mv krita_ai_diffusion-1.31.1/ai_diffusion krita_ai_diffusion-1.31.1/*.desktop . && \\" >> Dockerfile
	echo "	rm krita_ai_diffusion-1.31.1.zip && \\" >> Dockerfile
	echo "	rmdir krita_ai_diffusion-1.31.1 && \\" >> Dockerfile
	echo "	cd ai_diffusion && \\" >> Dockerfile
	echo "	python3 -m venv . && \\" >> Dockerfile
	echo "	./bin/python3 -m pip install aiohttp tqdm && \\" >> Dockerfile
	echo "	./bin/python3 download_models.py --sd15 --sdxl --checkpoints /comfyUI-server/ComfyUI" >> Dockerfile
	# echo "RUN mkdir /comfyUI-server/ComfyUI/models/inpaint" >> Dockerfile
	echo \ >> Dockerfile	
	echo "RUN cd /comfyUI-server/ComfyUI && \\" >> Dockerfile
	echo "	python3 -m venv . && \\" >> Dockerfile
	echo "	./bin/python3 -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu126 && ./bin/python3 -m pip install -r requirements.txt" >> Dockerfile
	echo \ >> Dockerfile

	echo "# CUSTOM_NODES" >> Dockerfile
	echo \ >> Dockerfile
	echo "RUN cd /comfyUI-server/ComfyUI/custom_nodes && \\" >> Dockerfile
	echo "	git clone https://github.com/Acly/comfyui-inpaint-nodes.git && \\" >> Dockerfile
	echo "	git clone https://github.com/Acly/comfyui-tooling-nodes.git && \\" >> Dockerfile
	echo "	git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git && \\" >> Dockerfile
	echo "	cd comfyui_controlnet_aux && \\" >> Dockerfile
	echo "	/comfyUI-server/ComfyUI/bin/python3 -s -m pip install -r requirements.txt" >> Dockerfile
	echo \ >> Dockerfile

	# IP-Adapter custom node
	echo "RUN cd /comfyUI-server/ComfyUI/custom_nodes/ && \\" >> Dockerfile
	echo "	git clone --depth 1 --single-branch https://github.com/cubiq/ComfyUI_IPAdapter_plus.git" >> Dockerfile

	echo \ >> Dockerfile

	# downloading ip-adapters

	# echo "RUN mkdir /comfyUI-server/ComfyUI/models/ipadapter" >> Dockerfile

	declare -A models_and_adapters=( 
        ["clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"
        
        ["clip_vision/CLIP-ViT-bigG-14-laion2B-39B-b160k.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors"
        
        ["clip_vision/clip-vit-large-patch14-336.bin"]="https://huggingface.co/Kwai-Kolors/Kolors-IP-Adapter-Plus/resolve/main/image_encoder/pytorch_model.bin"
                
        ["ipadapter/ip-adapter_sd15.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors"
        
        ["ipadapter/ip-adapter_sd15_light_v11.bin"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light_v11.bin"
        
        ["ipadapter/ip-adapter-plus_sd15.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors"
        
        ["ipadapter/ip-adapter-plus-face_sd15.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.safetensors"
        
        ["ipadapter/ip-adapter-full-face_sd15.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.safetensors"
        
        ["ipadapter/ip-adapter_sd15_vit-G.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_vit-G.safetensors"
        
        ["ipadapter/ip-adapter_sdxl_vit-h.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors"
        
        ["ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors"
        
        ["ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.safetensors"
        
        ["ipadapter/ip-adapter_sdxl.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.safetensors"

        ["upscale_models/4x_NMKD-Superscale-SP_178000_G.pth"]="https://huggingface.co/gemasai/4x_NMKD-Superscale-SP_178000_G/resolve/main/4x_NMKD-Superscale-SP_178000_G.pth"

        ["upscale_models/OmniSR_X2_DIV2K.safetensors"]="https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X2_DIV2K.safetensors"

        ["upscale_models/OmniSR_X3_DIV2K.safetensors"]="https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X3_DIV2K.safetensors"

        ["upscale_models/OmniSR_X4_DIV2K.safetensors"]="https://huggingface.co/Acly/Omni-SR/resolve/main/OmniSR_X4_DIV2K.safetensors"

        ["controlnet/control_v11p_sd15_inpaint_fp16.safetensors"]="https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors"

        ["controlnet/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"]="https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors"

        ["loras/Hyper-SDXL-8steps-CFG-lora.safetensors"]="https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-CFG-lora.safetensors"

        ["inpaint/fooocus_inpaint_head.pth"]="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/fooocus_inpaint_head.pth"

        ["inpaint/inpaint_v26.fooocus.patch"]="https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch"

        ["clip_vision/clip-vision_vit-h.safetensors"]="https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors"

        ["inpaint/MAT_Places512_G_fp16.safetensors"]="https://huggingface.co/Acly/MAT/resolve/main/MAT_Places512_G_fp16.safetensors"
    );

	for key in ${!models_and_adapters[@]}; do
		echo "RUN wget -O /comfyUI-server/ComfyUI/models/$key ${models_and_adapters[$key]}" >> Dockerfile
	done	

	echo \ >> Dockerfile
	# exposing and entrypoint	
	echo "EXPOSE 8188" >> Dockerfile
	echo "ENTRYPOINT [ \"/comfyUI-server/ComfyUI/bin/python3\", \"/comfyUI-server/ComfyUI/main.py\", \"--listen\" ]" >> Dockerfile
}

if [[ ! $(id) == *"root"* ]]; then
	echo "you should run this script as root user!";
	exit;
fi

nvidia_toolkit_installation
create_dockerfile