import numpy as np
from scipy.optimize import fsolve, minimize, Bounds

# Physical properties
rho_CH4 = 0.7168        # methane density (kg/m^3)
eta_generator = 0.3    # efficiency of gas turbine used for electricity generation
btu_h2o = 1500          # energy required to remove 1lb H2O (BTU/lb H2O)


# Conversions
t2lbs = 2205            # lbs to tonnes conversion factor (lb/tonne)
day2second = 86400      # seconds in a day (s/day)
acre2meter = 4046       # m^2 in 1 acre (m^2/acre)
joule2kwh = 3.6*1e6     # joules in 1 kWh (J/kWh)
ft3_m3 = 35.3           # cubic feet in a cubic meter (ft^3/m^3)
BTU_kWh = 3412          # BTU in a kWh (BTU/kWh)
SCF_BTU = 1000          # SCF in MMBTU (SCF/MMBTU)
lb_kg = 2.2             # lbs in kg (lb/kg)

# Prices
p = {}                  # price dictionary
p['land'] = 5890/1e6    # cost of land based on WI farmland average (MMUSD/acre)
p['bags'] = 5147/1e6    # b-PBR bag replacement cost (MMUSD/acres/yr)
p['inoc'] = 1590/1e6    # b-PBR inoculum and cultivation costs (MMUSD/acre/yr)
p['urea'] = 500/1e6     # urea cost (MMUSD/tonne)
p['mix'] = 3.3154e-5    # b-PBR mixing requirements (MMUSD/ha/day)
p['wat_del'] = 8.8e-8   # b-PBR power requirements for water delivery to units (MMUSD/ha/day)
p['cb'] = 7000/1e6      # CB price (MMUSD/tonne)
p['floc'] = 1e-4        # flocculation tank operating cost (MMUSD/tonne CB processed)
p['lam'] = 4.30e-7      # lamella clarifier operating cost (MMUSD/ tonne CB processed)
p['pf'] = 3.3e-8        # pressure filter operating energy demand (MMUSD/m^3 feed)
p['nat_gas'] = 5.84     # natural gas price (USD/MMBTU <--> USD/1000 SCF) 
p['elec'] = 0.11        # price of electricity (USD/kWh)
p['co2_rem'] = 40       # price of removing CO2 from biogas (USD/tonne CO2)
p['h2s_rem'] = 0.0667   # price of removing H2S from biogas (USD/kg biogas)
p['labor'] = 9.34/2428  # labor costs for PBR based on a facility of size 2428 acres (MMUSD/acre/yr)
p['maint'] = 0.05       # maintenance cost fraction (MMUSD/MMUSD ISBL)
p['op'] = 0.025         # business costs cost fraction (MMUSD/MMUSD ISBL)
p['ovhd'] = 0.05        # overhead costs cost fraction (MMUSD/MMUSD ISBL)
p['rin'] = 0.839        # RIN credit value for methane production (USD/kg CH4)
p['lcfs'] = 0.548       # LCFS credit value for metahne production (USD/kg CH4)
p['p_credit'] = 74.5    # P credit value for phosphorus capture (USD/kg P)
p['DAP_N'] = 1358.35    # Price of N when sourced from diammonium phosphate (USD/tonne)
p['DAP_P'] = 1487.71    # Price of P when sourced from diammonium phosphate (USD/tonne)


# Economic constants
## general project values
oprD = 365                          # operational days a year
CEPCI_2020 = 596.2                  # CEPCI for year project will be started
DROI = 0.15                         # discounted return on investment target of project
project_life = 15                   # project lifetime in years
gwp_electricity = 0.4795            # global warming potential of electricty use (tonne CO2/MWhr)

## digester and SLS
AD_CEPCI = 539.1                    # CEPCI for reference quoted AD system cost
SLS_CEPCI = 556.8                   # CEPCI for reference quoted SLS system cost

## bioreactors
PBR_scale_factor = 0.6              # scaling factor for purchase of PBRs
PBR_base_size = 5.08/acre2meter     # size of reference PBR system (acres)
PBR_base_price = 0.000279           # installed cost of reference PBR system (MMUSD)
PBR_CEPCI = 585.7                   # CEPCI for reference quoted PBR system cost

## separation train
FLOC_scale_factor = 0.6             # scaling factor for purchase of flocculation tank
FLOC_base_size = 266939             # size of reference flocculation tank (tonnes CB processed/yr)
FLOC_base_price = 0.1147            # installed cost of reference flocculation tank (MMUSD)
FLOC_CEPCI = 585.7                  # CEPCI for reference quoted flocculation tank cost

LAM_scale_factor = 0.6              # scaling factor for purchase of lamella clarifier
LAM_base_size = 266939              # size of reference lamella clarifier (tonnes CB processed/yr)
LAM_base_price = 2.5                # installed cost of reference lamella clarifier (MMUSD)
LAM_CEPCI = 585.7                   # CEPCI for reference quoted lamella clarifier cost

PF_scale_factor = 0.6               # scaling factor for purchase of pressure filter
PF_base_size = 17.76*365            # size of reference pressure filter (tonnes effluent/yr)
PF_base_price = 0.137               # installed cost of reference pressure filter (MMUSD)
PF_CEPCI = 381.8                    # CEPCI for reference quoted pressure filter cost

## thermal dryer
TD_scale_factor = 0.6               # scaling factor for purchase of thermal dryer
TD_base_size = 5.182*365            # size of reference thermal dryer (tonnes feed/yr)
TD_base_price = 0.706427            # installed cost of reference thermal dryer (MMUSD)
TD_CEPCI = 539.1                    # CEPCI for reference quoted thermal dryer cost

## cogeneration and amine
CGA_scale_factor = [0.8]            # scaling factor for cogeneration and amine unit
CGA_base_size = [620*365]           # size of reference cogeneration and amine unit (tonnes biogas feed/yr)
CGA_base_price = [13.1352]          # installed cost of reference cogeneration and amine unit (MMUSD)
CGA_CEPCI = [539.1, 444.2]          # CEPCI for reference quoted cogeneration and amine unit costs


# Front end parameters
## manure feed, source: https://doi.org/10.1016/j.scitotenv.2019.134059
M_in = 20832                            # tonnes/yr
P_in = M_in*0.078*7.8/1e3               # tonnes/yr Total P inlet manure, 7.8% TS, 7.8 g TP/kg dry
N_in = M_in*0.078*47/1e3                # tonnes/yr TotalN inlet manure, 7.8% TS, 47 g TN/kg dry
x_MinTS = 0.078                         # total solids fraction inlet manure, 7.8%
m_H2Oin = M_in*(1-x_MinTS)              # tonnes/yr Water inlet manure
x_digTS = 0.06                          # total solids in digestate, 6%
m_H2Oout = M_in*(1-x_digTS)             # tonnes/yr water in the digestate

## digestate, source: https://doi.org/10.1016/j.scitotenv.2019.134059
x_TP_dig = 1                            # mass frac of total P in the digestate
x_TN_dig = 1                            # mass frac of total N in the digestate
x_TAN_dig = 1                           # mass frac of N available in the digestate, 0.22%
x_solids_digestate = 0.08               # mass frac of solids in the digestate, 8%
x_TANin = 21                            # fraction of available N in manure that goes into AD digestate (kg N/tonne dry manure)

## biogas
y_MtoBG = 1/21                          # biogas production to manure feed ratio
x_CH4 = 0.6488                          # mass frac of CH4 in biogas
x_CO2 = 0.3488                          # mass frac of CO2 in biogas
x_H2S = 0.0024                          # mass frac of H2S in biogas
x_BG = np.array([x_CH4, x_CO2, x_H2S])  # composition of biogas stream
x_CH4market = min(1, 0.7055*1.50)       # fraction of generated methane sent to the market

## solids separator products, source: https://doi.org/10.1016/j.scitotenv.2019.134059
x_TS = 0.09                             # total solids fraction in digestate
x_TS_W = 0.47                           # mass fraction of water in the solid product, 47%
x_TS_P = 0.3                            # total P in the solid fraction, 30%
x_TS_N = 0.13                           # total N in the solid fration, 13%
x_av_P = 0.8                            # available P in the liquid fraction, 80%
x_av_N = 0.5                            # available N in the liquid fraction, 50%


# b-PBR parameters
"""
Dimension of X will depend on units used for m_v and Y_xv, recommend using SI units
"""
Y_xv = (0.00202*1e-6)*1.00              # Biomass production per photon consumed (kg*umol^-1)
eta = 0.23538*1.00                      # CB photon use efficiency (unit-less)
m_v = (917.5/3.6)*1.00                  # CB maintencance photon need (umol*kg^-1*s^-1)
pbr_NtoP = 7                            # CB N to P (N:P) ratio
X_0 = 0.03                              # Initial CB concentration (kg/m^3)
I_0 = 350*1.00                          # incident light intensity (umol*s^-1*m^-2)
t_batch =  3*86400*1.00                 # batch time (seconds)
SV_reactor = (4.48*1.22)/0.355*1.00     # Surface area to volume ration of bioreactors (m^-1)
sigma_a = 0.355/5.08                    # Bioreactor mass surface density of rack system (tonne/m^2)

x = np.array([I_0, SV_reactor, t_batch])

# Separation train and thermal dryer parameters
floc_req = 0#0.097                      # flocculant required on CB mass basis (tonnes floc/tonnes CB)
x_cbL = 0.016                           # mass frac of algae in lamella output
x_cbPF = 0.27                           # mass frac of algae in pressure filter output
x_cbTD = 1.0                            # mass frac of algae in dryer output
recycle_frac = 0.818179594328041        # recycle fraction of water in separation train; 0.95 for WT, 0.818179594328041 for mutant

#%% UNIT DEFINITIONS
# Digester
class DIGESTER():
    def __init__(self, Min, Pin, Nin, x_TANin, y_MtoBG, x_BG, x_solids_digestate, x_MinTS, x_TP_dig, x_TN_dig, x_TAN_dig, CEPCI):
        self.m_in = M_in                                # manure feed (tonnes/yr)
        self.m_TPin = P_in                              # P (total) in manure feed (tonnes/yr)
        self.m_TNin = N_in                              # N (total) in manure feed (tonnes/yr)
        self.x_TANin = x_TANin                          # fraction of total available N in solids (kg N/tonne dry manure)
        self.y_MtoBG = y_MtoBG                          # yield of biogas (tonnes biogas/tonnes manure)
        self.x_BG = x_BG                                # composition of biogas stream (CH4, CO2, and H2S)
        self.x_solids_digestate = x_solids_digestate    # solids fraction of digestate stream
        self.x_MinTS = x_MinTS                          # total solids fraction of manure feed       
        self.x_TP = x_TP_dig                            # fraction of total P (on mass basis) that remains in digestate (tonnes P dig/tonnes P feed)
        self.x_TN = x_TN_dig                            # fraction of total N (on mass basis) that remains in digestate (tonnes N dig/tonnes N feed)
        self.x_TAN = x_TAN_dig                          # fraction of N in digestate that is available N
        self.CEPCI = CEPCI                              # CECPI for year of reference system quote

    ## digester mass balance based on biogas yield (y_MtoBG) and composition (x_BG) values
    def mass_bal(self):
        self.m_BG = self.m_in*self.y_MtoBG                                  # biogas generation (tonnes/yr)
        self.m_D = self.m_in-self.m_BG                                      # digested manure outflow (tonnes/yr)
        self.m_CH4 = self.m_BG*self.x_BG[0]                                 # CH4 generated (tonnes/yr)
        self.m_CO2 = self.m_BG*self.x_BG[1]                                 # CO2 generated (tonnes/yr)
        self.m_H2S = self.m_BG*self.x_BG[2]                                 # H2S generated (tonnes/yr)
        self.m_TANin = self.m_in*self.x_MinTS*self.x_TANin/1e3              # mass flow of available N (tonnes/yr)
        self.m_OrgNin = self.m_TNin-self.m_TANin                            # mass flow of organic (unavailable) N (tonnes/yr) 
        self.m_TP_dig = self.m_TPin                                         # flow of P in outflow (tonnes/yr)
        self.m_TN_dig = self.m_TNin                                         # flow of N in outflow (tonnes/yr)
        self.m_TAN_dig = self.m_TANin*(self.x_TAN)                          # flow of avaialble N in outflow (tonnes/yr)                  <--- this is greater than m_TANin
        self.m_OrgN_dig = self.m_TN_dig-self.m_TAN_dig                      # flow of organic (unavailable) N in out outflow (tonnes/yr)
        self.m_H2O_dig = (1-self.x_solids_digestate)*self.m_D               # flow of water in outflow (tonnes/yr)
        self.error = self.m_in-self.m_BG-self.m_D
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around digester is not closed')

    ## digester costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = (937.1*self.m_in**0.6+75355)*(CEPCI_2020/self.CEPCI) # Capacity is in tonnes/yr
        return self.C/1e6
 
    
# Solids-liquids separator
class SLD_SEP():
    def __init__(self, m_in, P_in, N_in, W_in, x_TS_W, x_TS_P, x_TS_N, x_av_P, x_av_N, CEPCI):
        self.m_in = m_in            # feed into unit (tonnes/yr)
        self.P_in = P_in            # total P in feed (tonnes/yr)
        self.N_in = N_in            # total N in feed (tonnes/yr)
        self.W_in = W_in            # water in feed (tonnes/yr)
        self.x_TS = x_TS            # total solids mass fraction of feed
        self.x_TS_W = x_TS_W        # mass fraction of H2O in solids product
        self.x_TS_P = x_TS_P        # fraction of total P in solids product
        self.x_TS_N = x_TS_N        # fraction of total N solids product
        self.x_av_P = x_av_P        # fraction of available P
        self.x_av_N = x_av_N        # fraction of avialable N
        self.CEPCI = CEPCI          # CECPI for year of reference system quote

    ## digester mass balance based on biogas yield (y_MtoBG) and composition (x_BG) values
    def mass_bal(self):
        self.m_SSS = self.x_TS*self.m_in                # flow of solid product (tonnes/yr)
        self.m_PS = self.x_TS_P*self.P_in               # flow of P (total) in solid product(tonnes/yr)
        self.m_NS = self.x_TS_N*self.N_in               # flow of N (total) in solid product (tonnes/yr)
        self.m_H2OS = self.x_TS_W*self.m_SSS            # flow of H2O in solid product (tonnes/yr)
        self.m_SSL = (1-self.x_TS)*self.m_in            # flow of liquid product (tonnes/yr)
        self.m_PL = (1-self.x_TS_P)*self.P_in           # flow of P (total) in liquid product (tonnes/yr)
        self.m_NL = (1-self.x_TS_N)*self.N_in           # flow of N (total) in liquid product (tonnes/yr)
        self.m_Pav = self.x_av_P*self.m_PL              # flow of available P in liquid product (tonnes/yr)
        self.m_Nav = self.x_av_N*self.m_NL              # flow of available N in liquid product (tonnes/yr)
        self.m_H2OL= self.W_in - self.m_H2OS            # flow of H2O in liquid product (tonnes/yr)
        self.error = self.m_in-self.m_SSL-self.m_SSS
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around solids-liquids separator is not closed')

    ## digester costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.m_lb = t2lbs*self.m_in/oprD/24 # flow in lbs/hr
        self.C = (3.75*self.m_in+1786.9*np.log(self.m_lb)-9506.6)*(CEPCI_2020/self.CEPCI)
        return self.C/1e6


# PBR
class PBR():
    def __init__(self, X_0, Y_xv, eta, m_v, m_P, m_N, NtoP, m_CO2, m_H2O, sigma_a, scale_factor, base_size, base_price, CEPCI, fix_N = True):
        ## bioreactor constants
        self.X_0 = X_0              # initial CB concentration (kg/m^3)
        self.Y_xv = Y_xv            # biomass production per photon consumed (kg*umol^-1)
        self.eta = eta              # CB photon use efficiency (unit-less)
        self.m_v = m_v              # CB maintencance photon need (umol*kg^-1*s^-1)
        self.rho_eff = 1            # density of effluent (tonne/m^3)
        self.sigma_a = sigma_a      # areal density of PBR racks
        
        ## reactor feed
        self.m_P = m_P          # P in feed
        self.m_N = m_N          # total N in feed
        self.m_CO2 = m_CO2      # total CO2 in feed
        self.m_H2O = m_H2O      # total H2O in feed
        
        ## cb required for total P consumption
        self.P_dem = 0.144*1.00             # mass fraction of P in CB biomass; WT is 0.017, mutant is 0.144
        self.m_cb = self.m_P/self.P_dem     # total mass of CB required to consume all P
        
        ## supplemental N required
        if fix_N:
            self.NtoP = NtoP                        # desired N to P ratio in CB biomass
            self.N_dem = self.NtoP*self.P_dem       # N mass fraction in CB biomass at this ratio
        
        else:
            self.N_dem = 0.031                      # mass fraction of N in CB biomass; WT is 0.040, mutant is 0.031
        
        self.m_NR = self.N_dem*self.m_cb        # N required to achieve this ratio based on CB production
        self.m_NS = max(0, self.m_NR-self.m_N)  # supplemtal N that must be provided to achieve total N demand
        
        ## supplemental CO2 required
        self.x_Ccb = 0.234                                      # mass fraction of C in CB biomass; WT is 0.412, mutant is 0.234
        self.e_CO2 = 0.1                                        # excess CO2 to be fed to CB
        self.CO2_dem = (44/12)*(1+self.e_CO2)*(self.x_Ccb)      # mass of CO2 required per mass of CB produced
        self.m_CO2R = self.CO2_dem*self.m_cb                    # total mass of CO2 required achieve required CB production
        self.m_CO2S = max(0, self.m_CO2R-self.m_CO2)            # supplemental CO2 that must be provided to achieve CO2 demand
        
        ## economic parameters
        self.sf = scale_factor          # economies of scale nonlinear scale factor
        self.PBR0 = base_size           # size of reference system (acres)
        self.C0 = base_price            # purchase cost of reference system (MMUSD)
        self.CEPCI = CEPCI              # CECPI for year of reference system quote
        
        ## conversions and constants
        self.day2second = 86400         # seconds in a day
        self.oprD = 365                 # operational days a year
        self.acre2meter = 4046          # m^2 in an acre
        self.joule2kWh = 3.6*1e6        # joules in a kWh
        self.gPERL_tonnePERm3 = 1e3     # g/L to tonnes/m^3
        
        
    ## simulate CB growth according to Ryan Clark 2018 model
    def CB_GRO(self, x, lam = 655*1e-9, c = 3.0*1e8, h = 6.63*1e-34, A = 6.023*1e23):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            
        I0 = x[0, 0]        # incident light intensity
        SV = x[0, 1]        # bag surface area to volume ratio
        t_end = x[0, 2]     # batch time
        
        t = np.linspace(0, t_end, int(t_end)+1)
        kappa = self.Y_xv*self.m_v
        X_S = self.eta*(I0*SV)/self.m_v
        
        self.X = self.X_0*np.exp(-kappa*t)+X_S*(1-np.exp(-kappa*t))                                 # effluent titer (kg/m^3)
        self.m_out = self.m_cb/(self.X[-1]/self.gPERL_tonnePERm3)*self.rho_eff                      # reactor outflow (tonnes/yr)
        self.V = (t_end/self.day2second)*(self.m_cb/self.oprD)/(self.X[-1]/self.gPERL_tonnePERm3)   # reactor volume (m^3)
        self.SA = self.V*self.rho_eff/self.sigma_a/self.acre2meter                                  # reactor surface area (acres)
        
        
    ## reactor costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.SA/self.PBR0)**self.sf
        return self.C


# Flocculation tank
class FLOCCULATION_TANK(): # There should not be any separation happeining inside the flocculator
    def __init__(self, m_in, m_cb, floc_req, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in            # mass flow of feed (tonnes/yr)
        self.m_cb = m_cb            # mass flow of CB in feed (tonnes/yr)
        self.f_req = floc_req       # flocculant required (tonnes flocculant/tonnes CB)
        self.sf = scale_factor      # economies of scale nonlinear scale factor
        self.F0 = base_size         # size of reference system (tonnes CB/yr)
        self.C0 = base_price        # purchase cost of reference system (MMUSD)
        self.CEPCI = CEPCI          # CECPI for year of reference system quote
    
    ## flocculation tank mass balance based on addition of flocculant (still preliminary, might not do anything)
    def mass_bal(self):
        self.m_floc = self.m_cb*self.f_req              # mass flow of flocculant (tonnes/yr) (might not be used)
        self.m_BM = self.m_in+self.m_floc               # mass flow of effluent (tonnes/yr)
        self.error = self.m_BM-self.m_in-self.m_floc
    
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around flocculation tank is not closed')
    
    ## flocculation tank costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_cb/self.F0)**self.sf
        return self.C


# Lamella clarifier
class LAMELLA(): # The lamella is the clarifier separator
    def __init__(self, m_in, m_cb, x_BM_effl, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in            # mass flow of feed (tonnes/yr)
        self.m_cb = m_cb            # mass flow of CB in feed (tonnes/yr)
        self.x_cb = x_BM_effl       # mass fraction of CB in effluent (tonnes CB/tonnes effluent) (essentially a yield factor)
        self.sf = scale_factor      # economies of scale nonlinear scale factor
        self.L0 = base_size         # size of reference system (tonnes CB/yr)
        self.C0 = base_price        # purchase cost of reference system (MMUSD)
        self.CEPCI = CEPCI          # CECPI for year of reference system quote
        
    ## lamella clarifier mass balance based on effluent moisture content fraction 1-x_BM_effl
    def mass_bal(self):
        self.m_BM = self.m_cb/self.x_cb                 # mass flow of effluent (tonnes/yr)
        self.m_W = self.m_in-self.m_BM                  # recycled water (tonnes/yr)
        self.error = self.m_in-self.m_BM-self.m_W
    
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around lamella clarifier is not closed')

    ## lamella clarifier costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_cb/self.L0)**self.sf
        return self.C


class PRESSURE_FILTER():
    def __init__(self, m_in, m_cb, x_BM_effl, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in            # mass flow of feed (tonnes/yr)
        self.m_cb = m_cb            # mass flow of CB in feed (tonnes/yr)
        self.x_cb = x_BM_effl       # mass fraction of CB in effluent (tonnes CB/tonnes effluent) (essentially a yield factor)
        self.sf = scale_factor      # economies of scale nonlinear scale factor
        self.PF0 = base_size        # size of reference system (tonnes effluent/yr)
        self.C0 = base_price        # purchase cost of reference system (MMUSD)
        self.CEPCI = CEPCI          # CECPI for year of reference system quote

    ## pressure filter mass balance based on effluent moisture content x_BM_effl
    def mass_bal(self):
        self.m_BM = self.m_cb/self.x_cb                 # mass flow of effluent (tonnes/yr)
        self.m_W = self.m_in-self.m_BM                  # recycled water (tonnes/yr)
        self.error = self.m_in-self.m_BM-self.m_W
    
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around pressure filter is not closed')
            
    ## pressure filter costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_BM/self.PF0)**self.sf
        return self.C


# Thermal dryer
class THERMAL_DRYER():
    def __init__(self, m_in, m_cb, x_BM_effl, scale_factor, base_size, base_price, CEPCI):
        self.m_in = m_in            # mass flow of feed (tonnes/yr)
        self.m_cb = m_cb            # mass flow of CB in feed (tonnes/yr)
        self.x_cb = x_BM_effl       # mass fraction of CB in effluent (tonnes CB/tonnes effluent) (essentially a yield factor)
        self.sf = scale_factor      # economies of scale nonlinear scale factor
        self.TD0 = base_size        # size of reference system (tonnes feed/yr)
        self.C0 = base_price        # purchase cost of reference system (MMUSD)
        self.CEPCI = CEPCI          # CECPI for year of reference system quote
    
    ## thermal dryer mass balance based on specified final moisture content x_BM_effl
    def mass_bal(self):
        self.m_BM = self.m_cb/self.x_cb                 # mass flow of effluent (tonnes/yr)
        self.m_W = self.m_in - self.m_BM                # removed water (tonnes/yr)
        self.error = self.m_in-self.m_BM-self.m_W
    
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around thermal dryer is not closed')

    ## thermal dryer costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C = self.C0*(CEPCI_2020/self.CEPCI)*(self.m_in/self.TD0)**self.sf
        return self.C


# Co-gen and amine absorber
class COGEN_AMINE():
    def __init__(self, m_in, x_CH4market, Ct_Dig, scale_factor, base_size, base_price, CEPCI):
        self.m_CH4 = m_in[0]
        self.m_CO2 = m_in[1]
        self.m_H2S = m_in[2]
        self.x_CH4 = x_CH4market        # fraction of CH4 sold to market off-farm
        self.y_O2air = 0.21             # mol fraction of O2 in air
        self.Ct_Dig = Ct_Dig
        self.A0 = base_size[0]          # size of reference amine unit
        self.C0_A = base_price[0]       # installed cost of reference amine unit
        self.sf_A = scale_factor[0]     # scaling facotr for amine estimate
        self.CEPCI_CG = CEPCI[0]        # CEPCI of reference CG unit
        self.CEPCI_A = CEPCI[1]         # CEPCI of reference amine unit
    
    ## cogen and amine unit mass balance based on complete combustion of not exported methane
    ## CO2 and H2S in biogas stream are assumed to be completely removed in amine unit
    def mass_bal(self):
        self.m_Ain = self.m_CH4+self.m_CO2+self.m_H2S                       # mass flow of BG to amine scrubber
        self.m_Aout = self.m_CO2+self.m_H2S                                 # mass flow of gases exiting amine unit (CO2 and H2S)
        self.m_CH4M = self.m_CH4*self.x_CH4                                 # mass flow CH4 to market
        self.m_CH4CG = (1-self.x_CH4)*self.m_CH4                            # mass flow of CH4 into co-gen plant (tonnes/yr)
        self.m_CO2CG = self.m_CH4CG/16*44                                   # mass flow of CO2 produced in co-gen (tonnes/yr)
        self.m_H2OCG = self.m_CH4CG/16*2*18                                 # mass flow of H2O produced in co-gen (tonnes/yr)
        self.m_O2 = self.m_CH4CG/16*2*32                                    # required O2 (tonnes/yr)
        self.m_N2 = self.m_CH4CG/16*2*(1-self.y_O2air)/(self.y_O2air)*28    # mass flow of N2 in co-gen (tonnes/yr)
        self.m_flue = self.m_N2+self.m_H2OCG+self.m_CO2CG                   # mass flow of flue gas exiting co-gen (tonnes/yr)
        
        self.error = self.m_Ain+self.m_O2-(self.m_Aout+self.m_CH4M+self.m_CO2CG+self.m_H2OCG)
        
        if abs(self.error) > 1e-6:
            raise Exception('Mass balance around cogen and amine scrubbing unit is not closed')

    ## cogen and amine unit costing
    def econ(self, CEPCI_2020 = CEPCI_2020):
        self.C_CG = 0.67*self.Ct_Dig*(CEPCI_2020/self.CEPCI_CG)*(1-self.x_CH4)
        self.C_A = self.C0_A*(CEPCI_2020/self.CEPCI_A)*\
                  (self.m_Ain/self.A0)**self.sf_A
        return self.C_CG+self.C_A


#%% PROCESS SETUP AND SIMULATION
Digester = DIGESTER(M_in, P_in, N_in, x_TANin, y_MtoBG, x_BG, x_solids_digestate, x_MinTS, x_TP_dig, x_TN_dig, x_TAN_dig, AD_CEPCI)
Digester.mass_bal()
Ct_Digester = Digester.econ()

SLS = SLD_SEP(Digester.m_D, Digester.m_TP_dig, Digester.m_TN_dig, Digester.m_H2O_dig, x_TS_W, x_TS_P, x_TS_N, x_av_P, x_av_N, SLS_CEPCI)
SLS.mass_bal()
Ct_SLS = SLS.econ()

CGA = COGEN_AMINE([Digester.m_CH4, Digester.m_CO2, Digester.m_H2S], x_CH4market, Ct_Digester,
                  CGA_scale_factor, CGA_base_size, CGA_base_price, CGA_CEPCI)
CGA.mass_bal()
Ct_CGA = CGA.econ()

PBR_mod = PBR(X_0, Y_xv, eta, m_v, SLS.m_PL, SLS.m_Nav, pbr_NtoP, CGA.m_CO2, SLS.m_H2OL,
              sigma_a, PBR_scale_factor, PBR_base_size, PBR_base_price, PBR_CEPCI, fix_N = False)
PBR_mod.CB_GRO(x)
Ct_PBR = PBR_mod.econ()

Floc = FLOCCULATION_TANK(PBR_mod.m_out, PBR_mod.m_cb, floc_req,
                         FLOC_scale_factor, FLOC_base_size, FLOC_base_price, FLOC_CEPCI)
Floc.mass_bal()
Ct_FlocculationTank = Floc.econ()

Lam = LAMELLA(Floc.m_BM, Floc.m_cb, x_cbL,
              LAM_scale_factor, LAM_base_size, LAM_base_price, LAM_CEPCI)
Lam.mass_bal()
Ct_Lamella = Lam.econ()

PF = PRESSURE_FILTER(Lam.m_BM, Lam.m_cb, x_cbPF,
                     PF_scale_factor, PF_base_size, PF_base_price, PF_CEPCI)
PF.mass_bal()
Ct_PressureFilter = PF.econ()

TD = THERMAL_DRYER(PF.m_BM, PF.m_cb, x_cbTD,
                   TD_scale_factor, TD_base_size, TD_base_price, TD_CEPCI)
TD.mass_bal()
Ct_ThermalDryer = TD.econ()

m_recycle = (Lam.m_W+PF.m_W)*recycle_frac                   # recycle stream flow (tonnes/yr)
m_purge = (Lam.m_W+PF.m_W)*(1-recycle_frac)                 # purge stream flow (tonnes/yr)
m_makeup = (Floc.m_in-Floc.m_cb)-SLS.m_H2OL-m_recycle       # required makeup water (tonnes/yr)


#%% ECONOMICS CALCULATIONS
def MSP(msp, p, units, parameters, constants, DROI = 0.15, tax = 0.21):
    CGA = units[0]
    
    P_in = parameters[0]
    m_cb = parameters[1]
    depreciation = parameters[2]
    TCI = parameters[3]
    TOC = parameters[4]
    project_life = int(parameters[5])
    
    rho_CH4 = constants[0]
    ft3_m3 = constants[1]
    SCF_BTU = constants[2]
    BTU_kWh = constants[3]
    eta_generator = constants[4]
    
    x_credits = 0#1.57103702
    
    cb_revenue = msp*1e3*m_cb
    biogas_revenue = p['nat_gas']*((CGA.x_CH4*CGA.m_CH4*1e3/rho_CH4)*ft3_m3/SCF_BTU)
    electricity_revenue = p['elec']*eta_generator*((1-CGA.x_CH4)*CGA.m_CH4*1e3/rho_CH4)*ft3_m3*SCF_BTU/BTU_kWh
    
    p_credit_revenue = (74.5*x_credits)*P_in*1e3 
    rin_credits_revenue = 0.0*p['rin']*CGA.m_CH4*CGA.x_CH4*1e3
    lcfs_credits_revenue = 0.0*p['lcfs']*CGA.m_CH4*CGA.x_CH4*1e3
    
    REV = (cb_revenue+biogas_revenue+electricity_revenue+\
           p_credit_revenue+rin_credits_revenue+lcfs_credits_revenue)/1e6
    AATP = (1-tax)*(REV-TOC)+depreciation
    
    NPV = -TCI*np.ones(project_life+1)
    PVCF = NPV.copy()  # present value cashflow
    for i in range(1, project_life+1):
        PVCF[i] = AATP*(1+DROI)**(-i)
        NPV[i] = NPV[i-1]+PVCF[i]
    return NPV[-1].flatten()


# Capital Costs
ISBL = Ct_Digester+Ct_SLS+Ct_CGA+Ct_PBR+Ct_FlocculationTank+Ct_Lamella+Ct_PressureFilter+Ct_ThermalDryer
OSBL = 0.4*ISBL
ENG = (ISBL+OSBL)*0.3
CONT = (ISBL+OSBL)*0.2
LAND = p['land']*PBR_mod.SA
TCI = ISBL+OSBL+ENG+CONT+LAND
TCI_annualized = TCI*DROI/(1-(1+DROI)**-15)

# Operating costs
## fixed operating costs (FOC)
maintenance_costs = p['maint']*ISBL
operations_costs = p['op']*ISBL
overhead_costs = p['ovhd']*ISBL
depreciation = ISBL/project_life

## variable operating costs (VOC)
### front end
digester_cost = 0.096*Ct_Digester
sls_cost = (0.488*SLS.m_in+0.1*(1786.9*np.log(SLS.m_lb)-9506.6))/1e6
amine_cost = p['co2_rem']/1e6*CGA.m_CO2+p['h2s_rem']/1e6*Digester.m_BG*1000

### b-PBRs
pbr_replacement_cost = p['bags']*PBR_mod.SA
inoculum_cost = p['inoc']*PBR_mod.SA
mixing_cost = p['mix']*(PBR_mod.SA*acre2meter/1e4)*oprD
water_deliver_cost = p['wat_del']*(PBR_mod.SA*acre2meter/1e4)*oprD
urea_cost = p['urea']*PBR_mod.m_NS/(28/60)

### separation train and dryer
flocculation_tank_cost = p['floc']*Floc.m_cb
lamella_cost = p['lam']*Lam.m_cb
pressure_filter_cost = p['pf']*PF.m_in
dryer_cost = p['nat_gas']/1e6*(TD.m_W*1000*lb_kg*btu_h2o)/1e6

### labor
labor = p['labor']*PBR_mod.SA

### total operating costs
FOC = maintenance_costs+operations_costs+overhead_costs+depreciation
VOC = digester_cost+sls_cost+amine_cost+\
      pbr_replacement_cost+inoculum_cost+mixing_cost+water_deliver_cost+urea_cost+\
      flocculation_tank_cost+lamella_cost+pressure_filter_cost+dryer_cost+\
      labor
TOC = FOC+VOC

TAC = TCI_annualized+TOC # total annualized cost (USD/yr)

msp_args = (p,
            [CGA],
            np.array([PBR_mod.m_P+SLS.m_PS, PBR_mod.m_cb, depreciation, TCI, TOC, project_life]),
            np.array([rho_CH4, ft3_m3, SCF_BTU, BTU_kWh, eta_generator]))
msp = fsolve(MSP, 3, args = msp_args)

nutrient_value = (p['DAP_P']*PBR_mod.m_P)/PBR_mod.m_cb
msp2nv = msp*1000/nutrient_value
#%% CREDIT LEVEL CALCULATIONS
def credit_level(x, p, units, parameters, constants, DROI = 0.15, tax = 0.21, nutrient_value = nutrient_value):
    msp = nutrient_value/1000
    
    CGA = units[0]

    P_in = parameters[0]
    m_cb = parameters[1]
    depreciation = parameters[2]
    TCI = parameters[3]
    TOC = parameters[4]
    project_life = int(parameters[5])
    
    rho_CH4 = constants[0]
    ft3_m3 = constants[1]
    SCF_BTU = constants[2]
    BTU_kWh = constants[3]
    eta_generator = constants[4]
    
    
    cb_revenue = msp*1e3*m_cb
    biogas_revenue = p['nat_gas']*((CGA.x_CH4*CGA.m_CH4*1e3/rho_CH4)*ft3_m3/SCF_BTU)
    electricity_revenue = p['elec']*eta_generator*((1-CGA.x_CH4)*CGA.m_CH4*1e3/rho_CH4)*ft3_m3*SCF_BTU/BTU_kWh

    p_credit_revenue = (74.5*x)*P_in*1e3                          # 0.71642326 makes the msp:nv 1 (just from this alone); 0.34126446 all 3 credits
    rin_credits_revenue = 0.0*p['rin']*CGA.m_CH4*CGA.x_CH4*1e3    # 0.65169682 make the the msp:nv 1; 0.34126446 all 3 credits
    lcfs_credits_revenue = 0.0*p['lcfs']*CGA.m_CH4*CGA.x_CH4*1e3  # 0.65169682 make the the msp:nv 1; 0.34126446 all 3 credits

    REV = (cb_revenue+biogas_revenue+electricity_revenue+\
           p_credit_revenue+rin_credits_revenue+lcfs_credits_revenue)/1e6
    AATP = (1-tax)*(REV-TOC)+depreciation

    NPV = -TCI*np.ones(project_life+1)
    PVCF = NPV.copy()  # present value cashflow
    for i in range(1, project_life+1):
        PVCF[i] = AATP*(1+DROI)**(-i)
        NPV[i] = NPV[i-1]+PVCF[i]
    return np.abs(NPV[-1].flatten())


bounds = Bounds(np.zeros(1), 2*np.ones(1))
x0 = np.array([1.0])

sol = minimize(credit_level,
               x0 = x0,
               bounds = bounds,
               method = 'L-BFGS-B',
               args = msp_args)

x_credits = sol.x
#%% UTILITY USE AND GWP POTENTIAL
# Front end
digester_sls_utility = 0.17*eta_generator*(Digester.m_CH4*1e3/rho_CH4)*ft3_m3*SCF_BTU/BTU_kWh      # kWhr/yr electricity used
cga_utility = eta_generator*(Digester.m_CH4*(1-x_CH4market)*1e3/rho_CH4)*ft3_m3*SCF_BTU/BTU_kWh    # kWhr/yr electricity generated

# b-PBR
mixing_utility = mixing_cost*1e6/p['elec']                                                          # kWhr/yr electricity used
water_delivery_utility = water_deliver_cost*1e6/p['elec']                                           # kWhr/yr electricity used

# Dewatering train
flocculation_tank_utility = flocculation_tank_cost*1e6/p['elec']                                    # kWhr/yr electricity used
lamella_utility = lamella_cost*1e6/p['elec']                                                        # kWhr/yr electricity used
pressure_filter_utility = pressure_filter_cost*1e6/p['elec']                                        # kWhr/yr electricity used

# Thermal dryer
dryer_utility = (TD.m_W*1000*lb_kg*btu_h2o)/SCF_BTU                                                 # SCF/yr of natural gas used
dryer_energy = dryer_utility*SCF_BTU/BTU_kWh                                                        # energy delivered by natural gas

# Total process energy requirements (kWhr/yr)
process_energy = digester_sls_utility+mixing_utility+water_delivery_utility+\
                 flocculation_tank_utility+lamella_utility+pressure_filter_utility+\
                 dryer_energy-cga_utility
                 
# CO2 emissions from electricity consumption (tonnes CO2/yr)
electricity_co2 = gwp_electricity*(digester_sls_utility+mixing_utility+water_delivery_utility+
                                   flocculation_tank_utility+lamella_utility+pressure_filter_utility-
                                   cga_utility)/1e3

# CO2 emissions from natural consumption(tonnes CO2/yr)
naturalgas_co2 = (44/16)*(dryer_utility/ft3_m3)*rho_CH4/1e3

# Total CO2 emissions (tonnes CO2/yr)
gwp_total = electricity_co2+naturalgas_co2


print(msp2nv[0])
print(TCI)
print(TOC)
print(TAC)
print(process_energy/1000)
print(PBR_mod.SA)
print(m_makeup)
