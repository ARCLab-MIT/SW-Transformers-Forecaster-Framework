<h1 align="center">Solar and Geomagnetic Indices Forecasting Framework Using Transformers</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2406.15847">
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="190" height="20">
    <defs>
        <linearGradient id="gradientLeft" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#ff6b6b"/>
        <stop offset="100%" stop-color="#b21b1a"/>
        </linearGradient>
        <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
        <feOffset result="offOut" in="SourceAlpha" dx="0" dy="1" />
        <feGaussianBlur result="blurOut" in="offOut" stdDeviation="1" />
        <feBlend in="SourceGraphic" in2="blurOut" mode="normal" />
        </filter>
        <svg id="arXiv_logo" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 246.978 110.119"><path d="M492.976,269.5l24.36-29.89c1.492-1.989,2.2-3.03,1.492-4.723a5.142,5.142,0,0,0-4.481-3.161h0a4.024,4.024,0,0,0-3.008,1.108L485.2,261.094Z" transform="translate(-358.165 -223.27)" fill="#fff"/><path d="M526.273,325.341,493.91,287.058l-.972,1.033-7.789-9.214-7.743-9.357-4.695,5.076a4.769,4.769,0,0,0,.015,6.53L520.512,332.2a3.913,3.913,0,0,0,3.137,1.192,4.394,4.394,0,0,0,4.027-2.818C528.4,328.844,527.6,327.133,526.273,325.341Z" transform="translate(-358.165 -223.27)" fill="#fff"/><path d="M479.215,288.087l6.052,6.485L458.714,322.7a2.98,2.98,0,0,1-2.275,1.194,3.449,3.449,0,0,1-3.241-2.144c-.513-1.231.166-3.15,1.122-4.168l.023-.024.021-.026,24.851-29.448m-.047-1.882-25.76,30.524c-1.286,1.372-2.084,3.777-1.365,5.5a4.705,4.705,0,0,0,4.4,2.914,4.191,4.191,0,0,0,3.161-1.563l27.382-29.007-7.814-8.372Z" transform="translate(-358.165 -223.27)" fill="#fff"/><path d="M427.571,255.154c1.859,0,3.1,1.24,3.985,3.453,1.062-2.213,2.568-3.453,4.694-3.453h14.878a4.062,4.062,0,0,1,4.074,4.074v7.828c0,2.656-1.327,4.074-4.074,4.074-2.656,0-4.074-1.418-4.074-4.074V263.3H436.515a2.411,2.411,0,0,0-2.656,2.745v27.188h10.007c2.658,0,4.074,1.329,4.074,4.074s-1.416,4.074-4.074,4.074h-26.39c-2.659,0-3.986-1.328-3.986-4.074s1.327-4.074,3.986-4.074h8.236V263.3h-7.263c-2.656,0-3.985-1.329-3.985-4.074,0-2.658,1.329-4.074,3.985-4.074Z" transform="translate(-358.165 -223.27)" fill="#fff"/><path d="M539.233,255.154c2.656,0,4.074,1.416,4.074,4.074v34.007h10.1c2.746,0,4.074,1.329,4.074,4.074s-1.328,4.074-4.074,4.074H524.8c-2.656,0-4.074-1.328-4.074-4.074s1.418-4.074,4.074-4.074h10.362V263.3h-8.533c-2.744,0-4.073-1.329-4.073-4.074,0-2.658,1.329-4.074,4.073-4.074Zm4.22-17.615a5.859,5.859,0,1,1-5.819-5.819A5.9,5.9,0,0,1,543.453,237.539Z" transform="translate(-358.165 -223.27)" fill="#fff"/><path d="M605.143,259.228a4.589,4.589,0,0,1-.267,1.594L590,298.9a3.722,3.722,0,0,1-3.721,2.48h-5.933a3.689,3.689,0,0,1-3.808-2.48l-15.055-38.081a3.23,3.23,0,0,1-.355-1.594,4.084,4.084,0,0,1,4.164-4.074,3.8,3.8,0,0,1,3.718,2.656l14.348,36.134,13.9-36.134a3.8,3.8,0,0,1,3.72-2.656A4.084,4.084,0,0,1,605.143,259.228Z" transform="translate(-358.165 -223.27)" fill="#fff"/><path d="M390.61,255.154c5.018,0,8.206,3.312,8.206,8.4v37.831H363.308a4.813,4.813,0,0,1-5.143-4.929V283.427a8.256,8.256,0,0,1,7-8.148l25.507-3.572v-8.4H362.306a4.014,4.014,0,0,1-4.141-4.074c0-2.87,2.143-4.074,4.355-4.074Zm.059,38.081V279.942l-24.354,3.4v9.9Z" transform="translate(-358.165 -223.27)" fill="#fff"/><path d="M448.538,224.52h.077c1,.024,2.236,1.245,2.589,1.669l.023.028.024.026,46.664,50.433a3.173,3.173,0,0,1-.034,4.336l-4.893,5.2-6.876-8.134L446.652,230.4c-1.508-2.166-1.617-2.836-1.191-3.858a3.353,3.353,0,0,1,3.077-2.02m0-1.25a4.606,4.606,0,0,0-4.231,2.789c-.705,1.692-.2,2.88,1.349,5.1l39.493,47.722,7.789,9.214,5.853-6.221a4.417,4.417,0,0,0,.042-6.042L452.169,225.4s-1.713-2.08-3.524-2.124Z" transform="translate(-358.165 -223.27)" fill="#fff"/></svg>
    </defs>
        <rect id="leftRect" width="60" height="20" rx="5" ry="5" fill="url(#gradientLeft)" filter="url(#shadow)" style="clip-path: inset(0 5px 0 0);" />
        <rect id="rightRect" x="60" width="130" height="20" rx="5" ry="5" fill="#555" filter="url(#shadow)" style="clip-path: inset(0 0 0 5px);" />
        <rect id="separator" x="55" width="10" height="20" fill="#555" />
        <g fill="#fff" font-family="monospace" font-size="11">
        <use x="3" y="2" width="54" height="16" xlink:href="#arXiv_logo"/>
        <text id="textElement" x="65" y="15">physics:2406.15847</text>
        </g>
    </svg>
    </a>
</p>

This repository compiles the framework implementations developed so far, aiming to achieve **efficient and highly accurate** measurements of geomagnetic and solar indices. To meet these objectives, we propose a **transformer-based framework**.

Our primary focus is on **PatchTST** (Patch Time-Series Transformer), as proposed by [(Nie et al., 2022)](https://arxiv.org/pdf/2211.14730). PatchTST enhances time-series forecasting by dividing data into patches, effectively capturing local patterns. Its **channel-independent encoding** processes each variable separately, reducing interference between variables, while **transformer encoder layers** capture long-term dependencies. This architecture efficiently models both short- and long-term patterns while maintaining **computational efficiency**.

![FSMY 10.7 Solar Indices Comparison](.images/solfsmy_improvement.png)

The framework also includes a **custom loss function** that adjusts for imbalances in solar activity levels, improving accuracy during volatile periods. With an optimized **validation strategy**, it ensures strong generalization across various solar activity levels. Additionally, we have explored **Dst** and **Ap indices** predictions, using different approaches, such as combining them after preprocessing or decomposing them into trend, seasonal, and residual components.



## Key takeaways💡
* **Improved Accuracy**: The model achieves a 77% better MPE and 60% better SMPE than the SET benchmark, especially during high solar activity 🌞.

* **Custom Loss Function**: A tailored loss function balances solar activity levels, improving forecast accuracy for volatile periods 📊.

* **Efficient and Fast**: Patching and channel independence reduce memory usage, cut computational complexity, and enable training in under two minutes using dual GPUs ⚡💻.

* **Robust Validation**: A carefully crafted validation split ensures the model performs well across various solar activity levels, enhancing prediction reliability and the model ability of generalization🔍.

* **High Activity Forecasting**: The model excels during high solar activity, showing an 83% improvement in MPE 🚀.

## Setup ⚙️

To make sharing the code implementation easier and avoid dependency issues, we've used [DevContainers Tool](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) in Visual Studio Code. This tool deploys a Docker container with all the necessary dependencies, so you will need to install this code editor to run our code. If you're new to this environment, please follow [this tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) to set up your environment.

Additionally, we use [Weights and Biases (W&B)](https://wandb.ai/site) to manage different trainings and experiments. Much of the code depends on W&B functions, so we recommend using it for seamless code execution. You can follow their [tutorials](https://docs.wandb.ai/tutorials) to get started. If you prefer not to use W&B, make sure to **comment out** all lines where the W&B API is called.

Make sure the `.devcontainer/devcontainer.env` file exists because the build process will attempt to access it and throw an error if it can't be found. The file should be included in the repository you've downloaded but with a `.txt` extension. If you're using W&B, fill in the `devcontainer.env` file with the required information.

> [!IMPORTANT]  
> Before building the container, make sure you've **added the `.env` file** and that **Docker is running**.

Once everything is configured, you can build your project by pressing <kbd>Ctrl</kbd>+<kbd>⬆️</kbd>+<kbd>P</kbd> to open the command palette and selecting `Dev Containers: Build Project and Re-open inside a container`.

At this point, your environment should be set up and running.

> [!NOTE]  
> If you encounter issues during the container build, try removing the following arguments from the `devcontainers.json` file:
> ```json
> "${localEnv:EXISTS_GPU:--gpus}", "${localEnv:EXISTS_GPU:all}"
> ```
> This error may occur due to GPU detection issues. If you have a dedicated GPU, add these arguments instead:
> ```json
> "--gpus", "all"
> ```
> Otherwise, remove the mentioned arguments entirely.


## Contents 📚

Here we made a short description of the repository folder structure. Inside each of the main foilders you can also find some README.md files explaining its contents. 

- [`/dev_nbs`](/dev_nbs/): In this folder you can find all the code implementation refering to the data preparation, model training process, evaluation and hyperparameter tunning. Please for further information about this folder check its `README.md` file.

- [`/data`](/data/): This folder recopilates all the data that is being used inside the repository. In general, the files are automatically generated inside the code if they do not exist, however not all the data can be generated so please avoid deleting files without being sure they can be deleted. In case you want to update the data, you can change the `force_download` flag you can found in some of the configuration files, inside `dev_nbs`.

- [`/nbs`](/nbs/): This folder contains the implementation and explanation of all the functions needed for the code implementation, that will be reused in different parts of the code implementation. Please for further information about this folder check its `README.md` file.

> [!NOTE]
> We also use in our implementation [`nbdev`](https://github.com/fastai/nbdev) tool for development inside jupyter notebooks. For this reason inside the `nbs` folder we have the notebooks with explanations and test of different functions that then will be used in the code. However the final functions are exported using this tool into the `swdf` folder. To do so we make use of #|export tag, so please do not delete those tags if you improve the code.

- [`requirements.txt`](/requirements.txt): If you want to globally install dependencies inside the container, you must add them and rebuild the container using the command palette.


The other folders are not relevant for the code use.

## Acknowledgments 🗣️

Along all the repository we have used [`tsai`](https://github.com/timeseriesAI/tsai) library implemented by Iganacio Oguiza, so we want to give him special thanks for its work and some of the ideas here implemented. Furthermore, we specially acknoledge [Space Enviroment Technologies (SET)](https://spacewx.com/) for providing with a great portion of their data.


## Affiliation ✒️

This work has been implemented as a collaboration between members of the [Applied Intelligence and Data Analysis research group (AI+DA)](https://aida.etsisi.upm.es/) from Universidad Politécnica de Madrid (UPM) and researchers from [Astrodynamics, Space Robotics, and Controls Lab (ARCLab)](https://aeroastro.mit.edu/arclab/) from the Massachussets Institute of Technology (MIT).

<p align="center">
  <table style="margin: 0 auto;">
    <tr>
      <td align="center" style="padding-right: 50px;">
        <img src=".images/AIDA_logo.png" alt="AI+DA Logo" width="215"/><br/>
        <img src=".images/UPM_Logo.png" alt="UPM Logo" width="205"/>
      </td>
      <td align="center" style="padding-left: 50px;">
        <img src=".images/Lab-ARC-Logo-homepage.png" alt="ARCLab Logo" width="400"/><br/>
        <img src=".images/MIT_Logo.png" alt="MIT Logo" width="400"/>
      </td>
    </tr>
  </table>
</p>
