import streamlit as st
import numpy as np
import math
import cmath
from PIL import Image
import pandas as pd
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
from copy import copy, deepcopy
from matplotlib import pyplot as plt

st.set_page_config(page_title="Solving BVP", page_icon =":1234:", layout ="wide")


st.set_option('deprecation.showPyplotGlobalUse', False)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

st.markdown("""
    <style>
        body {
          background-color: #f7f3e9;
          padding: 20px;
          color: #444444;
          font-family: Arial, sans-serif;
        }
        div.stButton > button:first-child {
        background-color: #c47451;
        color: #ffffff;
        padding: 10px 20px;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        }
        div.stButton > button:first-child:hover {
        background-color: #a05a32;
        color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

def page1():
    with st.container():
        st.title("This is Earth-Deep Solver. Welcome!")
        st.subheader("This solver is designed to solve the boundary value problems for a thermoelastic half plane subjected to both heat sources and mass forces")
        st.write("Using this solver is simple - just input the necessary parameters and let the solver do the rest. You can input the material properties, the dimensions of the half plane, and the boundary conditions. The solver will then solve the partial differential equations governing the behavior of the half plane and provide you with the resulting temperature and displacement fields.")
        st.write("---")
        st.subheader("Creators")
        st.image(Image.open('logo of uni.png'), width=400)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Balzhan Shayakhmetova")
        with col2:
            st.write("Azamat Zhanbolat")
        with col3:
            st.write("Elmira Bukayeva")
        text = "Special acknowledgement to <span class='shiny-text'>Alipova B.N.</span>"
        css = """
            <style>
            .shiny-text {
                animation: shine 3s linear infinite;
            }

            @keyframes shine {
                0% { color: #1a1a1a; }
                25% { color: #333333; }
                50% { color: #FFFFFF; }
                75% { color: #333333; }
                100% { color: #c0c0c0; }
            }
            </style>
            """
        # Display the subheader with color change
        st.markdown(css+text, unsafe_allow_html=True)
    st.write("---")
    st.header("Get started!")
    st.write("##")
    tableRocksImage = Image.open('Tablerocks.png')
    st.subheader("The rock parameters you can choose from")
    st.image(tableRocksImage, use_column_width = True, caption='Table of rocks')
    st.subheader("BVPs")
    st.write("BVP stands for Boundary Value Problem, which refers to a type of mathematical problem where the solution is sought within a specific range and is determined by specifying boundary conditions at the endpoints of that range.")
    st.image(Image.open('BVPs.jpg'), width = 400, caption='BVPs')
    BVP_input = st.selectbox('Choose BVP', ('1', '2', '3', '4'))
    rock_parameters = st.selectbox("Choose the rock from the table or input your own parameters", ("granite", "sandstone", "silicified shale", "silistone", "shales", "other"))
    if rock_parameters:
        if rock_parameters == "granite" or rock_parameters == "sandstone" or rock_parameters == "silicified shale" or rock_parameters == "silistone" or rock_parameters == "shales":
            st.session_state.rock_parameters = rock_parameters
        else:
            sigmaPar = st.number_input("σ * 10^-3 [kg/m3]")
            youngsMod = st.number_input("E * 10^-10 [N/m2]")
            nuPar = st.number_input("ν")
            lambdaTPar = st.number_input("λ_t [Vt/m*K]")
            CTPar = st.number_input("C_t [J/N*K]")
            alphaTPar = st.number_input("α_t * 10^5 [I/K]")
            velPPar = st.number_input("ϑ_p [m/sec]")
            velSPar = st.number_input("ϑ_s [m/sec]")
            muPar = st.number_input("μ * 10^-10 [kg/m*sec^2]")
            lambdaPar = st.number_input("λ * 10^-10 [kg/m*sec^2]")
            gammaPar = st.number_input("γ * 10^-5 [Pa/K]")
            etaPar = st.number_input("η * 10^-8 [K*sec/m^2]")
            kappaPar = st.number_input("κ * 10^7[m^2/sec]")
            st.session_state.sigmaPar = sigmaPar
            st.session_state.youngsMod= youngsMod
            st.session_state.nuPar = nuPar
            st.session_state.lambdaTPar=lambdaTPar
            st.session_state.CTPar = CTPar
            st.session_state.alphaTPar = alphaTPar
            st.session_state.velPPar = velPPar
            st.session_state.velSPar = velSPar
            st.session_state.muPar = muPar
            st.session_state.lambdaPar = lambdaPar
            st.session_state.gammaPar = gammaPar
            st.session_state.etaPar = etaPar
            st.session_state.kappaPar = kappaPar
    omegaParameterFrom=st.number_input("Choose starting angular frequency ")
    omegaParameterTo = st.number_input("Choose ending angular frequency ")
    xiParInput = st.number_input("Choose constant xi")
    x_1 = st.number_input("Choose variable x1")
    x_2 = st.number_input("Choose variable x2")
    st.session_state.BVP_input = BVP_input
    st.session_state.omegaParameterFrom = omegaParameterFrom
    st.session_state.omegaParameterTo = omegaParameterTo
    st.session_state.xiParInput = xiParInput
    st.session_state.x_1 = x_1
    st.session_state.x_2 = x_2
    if st.button("Calculate", key="calculate_button"):
        st.session_state.page = "page2"


def page2():
    # Retrieve the user input from session state
    BVP_input = st.session_state.BVP_input
    rock_parameters = st.session_state.rock_parameters
    st.header("Visualization")
    st.write("for BVP ", BVP_input, " and for the ", rock_parameters)
    omegaParameterFrom = st.session_state.omegaParameterFrom
    omegaParameterTo = st.session_state.omegaParameterTo
    xiParInput = st.session_state.xiParInput
    x_1 = st.session_state.x_1
    x_2 = st.session_state.x_2
    if(BVP_input == "1"):
      alpha = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
      beta = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
    elif(BVP_input == "2"):
      alpha = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
      beta = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])
    elif(BVP_input == "3"):
      alpha = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]])
      beta = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 1]])
    elif(BVP_input == "4"):
      alpha = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 1]])
      beta = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 0]])
    else:
      st.write("You have to choose in range 1 to 4. Try again ...")
    rockParameters = pd.DataFrame({
        'granite': [2.61, 4.02, 0.26, 2.4, 946, 0.8, 5600, 2750, 1.974, 4.24, 13.3, 1.7, 9.27],
        'sandstone': [2.69, 4.13, 0.09, 1.66, 972, 0.64, 3272, 1293, 0.45, 1.98, 2.5, 0.44, 9.86],
        'silicified shale':[2.72, 5.2, 0.21, 2.4, 887, 0.53, 3243, 1808, 0.89, 1.08, 2.65, 0.32, 11.27],
        'silistone':[2.69, 3.87, 0.29, 1.49, 880, 0.5, 2390, 1204, 0.39, 0.756, 1.5, 0.29, 10.28],
        'shales':[2.77, 5.25, 0.15, 2.46, 866, 0.68, 4493, 2879, 2.296, 1, 5.2, 0.6, 9.46],
          }, index=['Density', 'Youngs modulus', 'Poisson ratio', 'Thermal conductivity', 'Heat capacity', 'Thermal expansion', 'Velocity of p waves', 'Velosity of s waves', 'Shear modulus', 'Lame parameter', 'Thermoelastic parameter gamma', 'Thermal expansion eta', 'Thermal conductivity kappa'])
    column_names_rocks = list(rockParameters.columns)
    column_names_rocks = [item.lower() for item in column_names_rocks]
    if rock_parameters.lower() in (item.lower() for item in column_names_rocks):
      sigmaPar = rockParameters.loc['Density', rock_parameters.lower()] #density of material
      youngsMod = rockParameters.loc['Youngs modulus', rock_parameters.lower()] #Young's modulus
      nuPar = rockParameters.loc['Poisson ratio', rock_parameters.lower()] #Poisson's ratio
      lambdaTPar = rockParameters.loc['Thermal conductivity', rock_parameters.lower()] #thermal conductivity
      CTPar = rockParameters.loc['Heat capacity', rock_parameters.lower()] #specific heat capacity
      alphaTPar = rockParameters.loc['Thermal expansion', rock_parameters.lower()] #thermal expansion coefficient
      velPPar = rockParameters.loc['Velocity of p waves', rock_parameters.lower()] #velocity of (compressional)p waves
      velSPar = rockParameters.loc['Velosity of s waves', rock_parameters.lower()] #velocity of shear waves
      muPar = rockParameters.loc['Shear modulus', rock_parameters.lower()] #shear modulus
      lambdaPar = rockParameters.loc['Lame parameter', rock_parameters.lower()] #the first Lame parameter
      gammaPar = rockParameters.loc['Thermoelastic parameter gamma', rock_parameters.lower()] #Gruneisen parameter
      etaPar = rockParameters.loc['Thermal expansion eta', rock_parameters.lower()] #coefficient of thermal expansion
      kappaPar = rockParameters.loc['Thermal conductivity kappa', rock_parameters.lower()] #thermal conductivity
      gammaBarPar = 0.94
      st.write("Successfully initialized parameters of ", rock_parameters, "!")
    else:
      sigmaPar = st.session_state.sigmaPar
      youngsMod = st.session_state.youngsMod
      nuPar = st.session_state.nuPar
      lambdaTPar = st.session_state.lambdaTPar
      CTPar = st.session_state.CTPar
      alphaTPar = st.session_state.alphaTPar
      velPPar = st.session_state.velPPar
      velSPar = st.session_state.velSPar
      muPar = st.session_state.muPar
      lambdaPar = st.session_state.lambdaPar
      gammaPar = st.session_state.gammaPar
      etaPar = st.session_state.etaPar
      kappaPar = st.session_state.kappaPar
      gammaPar = 0.94
      st.write("Successfully initialized input parameters")
    try:
        epsilonPar = (gammaPar* etaPar*kappaPar)/(lambdaPar + 2* etaPar)
        cPar_1 = (lambdaPar + 2*muPar)/sigmaPar
        cPar_2 = muPar/sigmaPar
        xiPar = 1
        xiBarPar = xiPar/sigmaPar
        n = np.array([1, 0, 0])

        def Kronecker(i,j): #Kronecker's delta
          if i==j:
            return 1
          else:
            return 0

        def D_iz():
            Diz = np.zeros((3, 3), dtype = 'complex_')
            for i in range(3):
                for z in range(3):
                  Diz[i, z] += muPar * n[z]
                  Diz[i, z] += -gammaPar * n[i] * Kronecker(2, z)
                  for j in range(3):
                      Diz[i, z] += lambdaPar * n[i] * Kronecker(j, z)
                      Diz[i, z] += muPar * n[j] * Kronecker(z, i)
            return Diz

        def B():
          B = np.zeros((3, 3), dtype = 'complex_')
          for z in range(3):
              for j in range(3):
                  B[z,j] = alpha[z,j] + alpha[z,2] * Kronecker(j,2) + beta[z,2] * Kronecker(j,2)
                  for l in range(3):
                      B[z,j] += beta[z,l] * D_iz()[l, j]
          return B

        def root_K1(xiPar, omegaPar):
          rootK1_sq = math.pow(xiPar, 2) - (math.pow(omegaPar, 2)/cPar_2)
          return rootK1_sq

        def root_K2(xiPar, omegaPar):
          temp1 = omegaPar**2/cPar_1 + 1j*omegaPar*(1+epsilonPar)/kappaPar
          temp2 = (omegaPar**2/cPar_1 + 1j*omegaPar*(1+epsilonPar)/kappaPar)**2 - 4j*omegaPar**3/(kappaPar*cPar_1)
          rootK2_sq = xiPar**2 - 0.5*(temp1 + np.sqrt(temp2))
          return rootK2_sq

        def root_K3(xiPar, omegaPar):
          temp1 = omegaPar**2/cPar_1 + 1j*omegaPar*(1+epsilonPar)/kappaPar
          temp2 = (omegaPar**2/cPar_1 + 1j*omegaPar*(1+epsilonPar)/kappaPar)**2 - 4j*omegaPar**3/(kappaPar*cPar_1)
          rootK3_sq = xiPar**2 - 0.5*(temp1 - np.sqrt(temp2))
          return rootK3_sq

        def plottingK1(xiPar):
          omegaChange = np.linspace(omegaParameterFrom, omegaParameterTo, 10)
          KRoot1Change = np.zeros((10), dtype = "complex")
          for i in range(len(omegaChange)):
            KRoot1Change[i] = root_K1(xiPar, omegaChange[i])
          fig, ax = plt.subplots(figsize=(6, 4))
          ax.plot(np.real(KRoot1Change), np.imag(KRoot1Change), '-.r*')
          ax.set_xlabel('Real part')
          ax.set_ylabel('Imaginary part')
          ax.set_title('root_K1(xiPar={}, omega)'.format(xiPar))
          plt.savefig('plot1.png')
          plt.close()
          plot1 = Image.open('plot1.png')
          st.image(plot1, width=400)

        def plottingK2(xiPar):
          omegaChange = np.linspace(omegaParameterFrom, omegaParameterTo, 10)
          KRoot2Change = np.zeros((10), dtype = "complex")
          for i in range(len(omegaChange)):
            KRoot2Change[i] = root_K2(xiPar, omegaChange[i])
          fig, ax = plt.subplots(figsize=(6, 4))
          ax.plot(np.real(KRoot2Change), np.imag(KRoot2Change), '-.r*')
          ax.set_xlabel('Real part')
          ax.set_ylabel('Imaginary part')
          ax.set_title('root_K2(xiPar={}, omega)'.format(xiPar))
          plt.savefig('plot2.png')
          plt.close()
          plot1 = Image.open('plot2.png')
          st.image(plot1, width=400)

        def plottingK3(xiPar):
          omegaChangeRoot3 = np.linspace(omegaParameterFrom, omegaParameterTo, 10)
          KRoot3Change = np.zeros((10), dtype = "complex")
          for i in range(len(omegaChangeRoot3)):
            KRoot3Change[i] = root_K3(xiPar, omegaChangeRoot3[i])
          fig, ax = plt.subplots(figsize=(6, 4))
          ax.plot(np.real(KRoot3Change), np.imag(KRoot3Change), '-.r*')
          ax.set_xlabel('Real part')
          ax.set_ylabel('Imaginary part')
          ax.set_title('root_K3(xiPar={}, omega)'.format(xiPar))
          plt.savefig('plot3.png')
          plt.close()
          plot1 = Image.open('plot3.png')
          st.image(plot1, width=400)

        col1, col2, col3 = st.columns(3)
        with col1:
            plottingK1(xiParInput)

        with col2:
            plottingK2(xiParInput)

        with col3:
            plottingK3(xiParInput)

        def kRootList(xiPar, omegaChange):
            kRootList = np.array([root_K1(xiPar, omegaChange), root_K2(xiPar, omegaChange), root_K3(xiPar, omegaChange)])
            return kRootList

        def vListTotal(xiPar, omegaChange):
              vListTotal = np.zeros((3, 3),dtype = 'complex_')
              v1Element = np.zeros((3),dtype = 'complex_')
              v2Element = np.zeros((3),dtype = 'complex_')
              v3Element = np.array([1, 1, 1])
              DeltaMatrixDetJList = np.zeros((3),dtype = 'complex_')
              DeltaMatrixDetJ1List = np.zeros((3),dtype = 'complex_')
              DeltaMatrixDetJ2List = np.zeros((3),dtype = 'complex_')
              for j in range(3):
               DeltaMatrixJ = np.array([[cPar_1 * kRootList(xiPar, omegaChange)[j]
                                          + omegaChange**2 - cPar_2 * xiPar**2, 1j
                                          * (cPar_1 - cPar_2) * xiPar
                                          * np.sqrt(kRootList(xiPar, omegaChange)[j])],
                                          [1j * (cPar_1 - cPar_2) * xiPar
                                           * np.sqrt(kRootList(xiPar, omegaChange)[j]),
                                           omegaChange**2 - cPar_1 * xiPar**2 + cPar_2
                                           * kRootList(xiPar, omegaChange)[j]]])
               DeltaMatrixJ1 = np.array([[gammaBarPar *
                                           np.sqrt(kRootList(xiPar, omegaChange)[j]),
                                           1j * (cPar_1 - cPar_2) * xiPar *
                                           np.sqrt(kRootList(xiPar, omegaChange)[j])],
                                          [1j * gammaBarPar * xiPar, omegaChange**2
                                           - cPar_1*xiPar**2
                                           + cPar_2*kRootList(xiPar, omegaChange)[j]]])
               DeltaMatrixJ2 = np.array([[cPar_1 * kRootList(xiPar, omegaChange)[j]
                                           + omegaChange**2 - cPar_2*xiPar**2, gammaBarPar
                                           * np.sqrt(kRootList(xiPar, omegaChange)[j])],
                                          [1j * (cPar_1 - cPar_2)* xiPar *
                                           np.sqrt(kRootList(xiPar, omegaChange)[j]),
                                           1j * gammaBarPar * xiPar]])
               DeltaMatrixDetJList[j] = np.linalg.det(DeltaMatrixJ)
               DeltaMatrixDetJ1List[j] = np.linalg.det(DeltaMatrixJ1)
               DeltaMatrixDetJ2List[j] = np.linalg.det(DeltaMatrixJ2)
              for j in range(3):
                v1Element[j] = DeltaMatrixDetJ1List[j]/DeltaMatrixDetJList[j]
                v2Element[j] = DeltaMatrixDetJ2List[j]/DeltaMatrixDetJList[j]
              vListTotal = np.array([v1Element, v2Element, v3Element])
              return vListTotal

        def b_lj(xiPar, omegaChange):
              b_ljMatrix = np.zeros((3,3),dtype = 'complex_')
              for k in range(3):
                for l in range(3):
                  for j in range(3):
                    b_ljMatrix[l, j] +=B()[l, k] * vListTotal(xiPar, omegaChange)[k, j]
              return b_ljMatrix

        def a_lj(j, x_1, xiPar, omegaChange):
              a = deepcopy(b_lj(xiPar, omegaChange))
              new_col = np.zeros((3),dtype = 'complex_')
              new_col[0] = Kronecker(j, 0)
              new_col[1] = Kronecker(j, 1)
              new_col[2] = Kronecker(j, 2)
              new_col = new_col.T
              a[:, j] = new_col
              return a

        def ATensor(x_1, xiPar, omegaChange):
              ATensor = np.zeros((3),dtype = 'complex_')
              for j in range(3):
                ATensor[j] = np.linalg.det(a_lj(j, x_1, xiPar, omegaChange))/np.linalg.det(b_lj(xiPar, omegaChange))
              return ATensor

        def calculate_VTensor(x_1, x_2,xiPar, omegaChange):
              ATensorV = ATensor(x_1, xiPar, omegaChange)
              vTotalList = vListTotal(xiPar, omegaChange)
              kList = kRootList(xiPar, omegaChange)
              VTensor = np.zeros((3, len(kList)), dtype=np.complex128)
              VTensorElement = complex(0, 0)
              for m in range(3):
                for k in range(len(kList)):
                  VTensorElement = complex(0, 0)
                  for j in range(3):
                    integrand = lambda xiPar: ATensorV[j] * vTotalList[k, j] * np.exp(1j * kList[j] * x_1 - 1j * xiPar * x_2)
                    VTensorElement += quad(integrand, -np.inf, np.inf)[0]
                  VTensorElement *= 1 / (2 * np.pi)
                  VTensor[k, m] = VTensorElement/np.exp(100)
              return VTensor

        def F_S(x_1, x_2):
            return np.heaviside(x_1, x_2)

        def Q_S(x_1, x_2):
            return np.heaviside(x_1, x_2)

        def calculate_ui(x_1, x_2, xiPar, omegaChange):
            VTensor = calculate_VTensor(x_1, x_2, xiPar, omegaChange)

            u = np.zeros(2, dtype=np.complex128)
            for i in range(2):
                u[i] = np.convolve(VTensor[i, 0], F_S(x_1, x_2), mode = 'valid') + np.convolve(VTensor[i, 1], F_S(x_1, x_2), mode='valid') + np.convolve(VTensor[i, 2], Q_S(x_1, x_2), mode='valid')
            return u

        def calculate_theta(x_1, x_2, xiPar, omegaChange):
            VTensor = calculate_VTensor(x_1, x_2, xiPar, omegaChange)
            theta = np.convolve(VTensor[2, 0],  F_S(x_1, x_2), mode='valid') + np.convolve(VTensor[2, 1], F_S(x_1, x_2), mode='valid') + np.convolve(VTensor[2, 2], Q_S(x_1, x_2), mode='valid')
            return theta

        def plot_u_theta(x_1, x_2, xiPar, omegaChange):
          x_1_values = np.linspace(1, 10, 10)
          x_2_values = np.linspace(1, 10, 10)
          xiParChange = np.linspace(1, 10, 10)
          omega_values = np.linspace(omegaParameterFrom, omegaParameterTo, 10)
          # Initialize arrays to store the results for xi
          u_values_xi = np.zeros((10, 2), dtype=np.complex128)
          theta_values_xi = np.zeros(10, dtype=np.complex128)
          # Initialize arrays to store the results for x_1
          u_values_x1 = np.zeros((10, 2), dtype=np.complex128)
          theta_values_x1 = np.zeros(10, dtype=np.complex128)
          # Initialize arrays to store the results for x_2
          u_values_x2 = np.zeros((10, 2), dtype=np.complex128)
          theta_values_x2 = np.zeros(10, dtype=np.complex128)
          # Initialize arrays to store the results for omega
          u_values_omega = np.zeros((10, 2), dtype=np.complex128)
          theta_values_omega = np.zeros(10, dtype=np.complex128)
          # Iterate over each x_1 value and calculate u and theta
          for i, x_1 in enumerate(x_1_values):
              u_values_x1[i] = calculate_ui(x_1, x_2, xiPar, omegaChange)
              theta_values_x1[i] = calculate_theta(x_1, x_2, xiPar, omegaChange)
          # Iterate over each x_2 value and calculate u and theta
          for i, x_2 in enumerate(x_2_values):
                u_values_x2[i] = calculate_ui(x_1, x_2, xiPar, omegaChange)
                theta_values_x2[i] = calculate_theta(x_1, x_2, xiPar, omegaChange)
          # Iterate over each x_2 value and calculate u and theta
          for i, xi in enumerate(xiParChange):
              u_values_xi[i] = calculate_ui(x_1, x_2, xi, omegaChange)
              theta_values_xi[i] = calculate_theta(x_1, x_2, xi, omegaChange)
          for i, omega in enumerate(omega_values):
              u_values_omega[i] = calculate_ui(x_1, x_2, xi, omega)
              theta_values_omega[i] = calculate_theta(x_1, x_2, xi, omega)
          # Plot the graphs
          plt.figure(figsize=(10, 5))

          # Plot u for x_1
          plt.subplot(2, 2, 1)
          plt.plot(x_1_values, u_values_x1[:, 0].real, 'r', label ='u_1')
          plt.plot(x_1_values, u_values_x1[:, 1].real, 'b', label='u_2')
          plt.xlabel('x_1')
          plt.ylabel('u')
          plt.title('Graphs of u for x_1')
          plt.legend()

          # Plot theta for x_1
          plt.subplot(2, 2, 2)
          plt.plot(x_1_values, theta_values_x1.real, 'g', label = 'theta_1')
          plt.xlabel('x_1')
          plt.ylabel('theta')
          plt.title('Graph of theta for x_1')
          # Show the plots
          plt.tight_layout()
          plt.savefig("plot_image1.png")
          plot1 = Image.open("plot_image1.png")
          st.image(plot1, width=800)


          # Plot u for x_2
          plt.subplot(2, 2, 3)
          plt.plot(x_2_values, u_values_x2[:, 0].real, 'r', label ='u_1')
          plt.plot(x_2_values, u_values_x2[:, 1].real, 'b', label='u_2')
          plt.xlabel('x_2')
          plt.ylabel('u')
          plt.title('Graphs of u for x_2')
          plt.legend()

          # Plot theta for x_2
          plt.subplot(2, 2, 4)
          plt.plot(x_2_values, theta_values_x2.real, 'g', label = 'theta_2')
          plt.xlabel('x_2')
          plt.ylabel('theta')
          plt.title('Graph of theta for x_2')

          # Show the plots
          plt.tight_layout()
          plt.savefig("plot_image2.png")
          plot2 = Image.open("plot_image2.png")
          st.image(plot2, width=800)

          #Plot u for xi
          plt.subplot(1, 2, 1)
          plt.plot(xiParChange, u_values_xi[:, 0].real,'r', label='u_1')
          plt.plot(xiParChange, u_values_xi[:, 1].real, 'g', label='u_2')
          plt.xlabel('xi')
          plt.ylabel('u')
          plt.title('Graphs of u')
          plt.legend()

           # Plot theta for xi
          plt.subplot(1, 2, 2)
          plt.plot(xiParChange, theta_values_xi.real, 'g')
          plt.xlabel('xi')
          plt.ylabel('theta')
          plt.title('Graph of theta')

           # Show the plots
          plt.tight_layout()
          plt.savefig("plot_image3.png")
          plot3 = Image.open("plot_image3.png")
          st.image(plot3, width=800)

          #Plot u for omega
          plt.subplot(1, 2, 1)
          plt.plot(omega_values, u_values_omega[:, 0].real,'r', label='u_1')
          plt.plot(omega_values, u_values_omega[:, 1].real, 'g', label='u_2')
          plt.xlabel('omega')
          plt.ylabel('u')
          plt.title('Graphs of u')
          plt.legend()

           # Plot theta for xi
          plt.subplot(1, 2, 2)
          plt.plot(xiParChange, theta_values_omega.real, 'g')
          plt.xlabel('omega')
          plt.ylabel('theta')
          plt.title('Graph of theta')

           # Show the plots
          plt.tight_layout()
          plt.savefig("plot_image4.png")
          plot4 = Image.open("plot_image4.png")
          st.image(plot4, width=800)


        col4, col5, col6 = st.columns(3)
        with col4:
            st.write("B matrix")
            st.table(B())

        with col5:
            st.write("D matrix")
            st.table(D_iz())

        with col6:
            st.write("Roots of characteristic equation")
            st.table(kRootList(xiParInput, omegaParameterTo-omegaParameterFrom))
        st.write("#")
        with col4:
            st.write("Total ϑ list", vListTotal(xiParInput, omegaParameterTo-omegaParameterFrom).astype('object'))

        with col5:
            st.write("small b matrix", b_lj(xiParInput, omegaParameterTo-omegaParameterFrom))

        with col6:
            st.write("A tensor", ATensor(1, xiParInput, omegaParameterTo-omegaParameterFrom))

        st.write("The final V Tensor is", calculate_VTensor(x_1, x_2, xiParInput, omegaParameterTo-omegaParameterFrom))

        col7, col8 = st.columns(2)
        with col7:
            st.write("Displacement u", calculate_ui(x_1, x_2, xiParInput, omegaParameterTo-omegaParameterFrom))
        with col8:
            st.write("Temperature theta", calculate_theta(x_1, x_2, xiParInput, omegaParameterTo-omegaParameterFrom))
        st.write("Solution of stationary oskilations f_z(x_2)")
        plot_u_theta(x_1, x_2, xiParInput, omegaParameterTo-omegaParameterFrom)
    except NameError:
        st.warning("No operations to perform")
    if st.button( "Go back to solver"):
        st.session_state.page = "page1"

# Define the main function
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "page1"
    # Display the current page
    if st.session_state.page == "page1":
        page1()
    elif st.session_state.page == "page2":
        page2()

# Run the app
if __name__ == "__main__":
    main()
