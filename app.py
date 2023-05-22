import streamlit as st
import numpy as np
import math
import cmath
import math
from PIL import Image
import pandas as pd
import scipy.integrate as integrate
from scipy.integrate import quad
import scipy.special as special
from copy import copy, deepcopy
from matplotlib import pyplot as plt

st.set_page_config(page_title="Solving BVP", page_icon =":tada:", layout ="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style/style.css")

def page1():
    st.title("This is Earth-Deep Solver. Welcome!")
    st.subheader("This solver is designed to solve the boundary value problems for a thermoelastic half plane subjected to both heat sources and mass forces")
    st.write("Using this solver is simple - just input the necessary parameters and let the solver do the rest. You can input the material properties, the dimensions of the half plane, and the boundary conditions. The solver will then solve the partial differential equations governing the behavior of the half plane and provide you with the resulting temperature and displacement fields.")

    st.write("---")
    st.header("Get started!")
    st.write("##")
    tableRocksImage = Image.open('Tablerocks.png')
    st.subheader("The rock parameters you can choose from")
    st.image(tableRocksImage, width=800, caption='Table of rocks')
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
    omegaParameterFrom=st.number_input("Choose frequency 1 ")
    omegaParameterTo = st.number_input("Choose frequency 2 ")
    xiParInput = st.number_input("Choose constant xi")
    st.session_state.BVP_input = BVP_input
    st.session_state.omegaParameterFrom = omegaParameterFrom
    st.session_state.omegaParameterTo = omegaParameterTo
    st.session_state.xiParInput = xiParInput
    if st.button("Calculate"):
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
                  B[z,j] = alpha[z,j] + alpha[z,2] * Kronecker(j,2)
                  + beta[z,2] * Kronecker(j,2)
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
          for i in range(len(omegaChange)-1):
            KRoot1Change[i] = root_K1(xiPar, omegaChange[i])
          fig, ax = plt.subplots(figsize=(6, 4))
          ax.plot(np.real(KRoot1Change), np.imag(KRoot1Change), '-.r*')
          ax.set_xlabel('Real part')
          ax.set_ylabel('Imaginary part')
          ax.set_title('root_K1(xiPar={}, omega)'.format(xiPar))
          plt.savefig('plot.png')
          plt.show()

        def plottingK2(xiPar):
          omegaChange = np.linspace(omegaParameterFrom, omegaParameterTo, 10)
          KRoot2Change = np.zeros((10), dtype = "complex")
          for i in range(len(omegaChange)-1):
            KRoot2Change[i] = root_K2(xiPar, omegaChange[i])
          fig, ax = plt.subplots(figsize=(3, 3))
          ax.plot(np.real(KRoot2Change), np.imag(KRoot2Change), '-.r*')
          ax.set_xlabel('Real part')
          ax.set_ylabel('Imaginary part')
          ax.set_title('root_K2(xiPar={}, omega)'.format(xiPar))
          plt.savefig('plot.png')
          plt.show()

        def plottingK3(xiPar):
          omegaChangeRoot3 = np.linspace(omegaParameterFrom, omegaParameterTo, 10)
          KRoot3Change = np.zeros((10), dtype = "complex")
          for i in range(len(omegaChangeRoot3)-1):
            KRoot3Change[i] = root_K3(xiPar, omegaChangeRoot3[i])
          fig, ax = plt.subplots(figsize=(3, 3))
          ax.plot(np.real(KRoot3Change), np.imag(KRoot3Change), '-.r*')
          ax.set_xlabel('Real part')
          ax.set_ylabel('Imaginary part')
          ax.set_title('root_K3(xiPar={}, omega)'.format(xiPar))
          plt.savefig('plot.png')
          plt.show()

        st.pyplot(plottingK1(xiParInput))
        st.pyplot(plottingK2(xiParInput))
        st.pyplot(plottingK3(xiParInput))

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
              for j in range(3):
                new_col[j] = Kronecker(j, 1)
              new_col = new_col.T
              a[:, j] = new_col
              return a

        def ATensor(x_1, xiPar, omegaChange):
              ATensor = np.zeros((3),dtype = 'complex_')
              for j in range(3):
                ATensor[j] = np.linalg.det(a_lj(j, x_1, xiPar, omegaChange))/np.linalg.det(b_lj(xiPar, omegaChange))
              return ATensor


        #np.set_printoptions(linewidth=np.inf)
        st.write(B())
        st.write("D matrix", D_iz())
        st.write("Roots of characteristic equation", kRootList(xiParInput, omegaParameterTo-omegaParameterFrom).astype('object'))
        st.write("Total V list", vListTotal(xiParInput, omegaParameterTo-omegaParameterFrom).astype('object'))
        #print(b_lj(xiParInput, omegaParameterTo-omegaParameterFrom))
        #st.write("A tensor", ATensor(1, xiParInput, omegaParameterTo-omegaParameterFrom))
    except NameError:
        st.warning("No operations to perform")
    if st.button("Go back to page 1"):
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
