# For downloading data
import numpy


# TODOs: 
# Having OI, P, R, and RINFO in the argument list will cause error. 
def myTradingSystem(DATE, OPEN, HIGH, LOW, CLOSE, VOL, # OI, P, R, RINFO,  
                    USA_ADP, USA_EARN, USA_HRS, USA_BOT, USA_BC, USA_BI, 
                    USA_CU, USA_CF, USA_CHJC, USA_CFNAI, USA_CP, USA_CCR, USA_CPI, USA_CCPI, USA_CINF, 
                    USA_DFMI, USA_DUR, USA_DURET, USA_EXPX, USA_EXVOL, USA_FRET, USA_FBI, USA_GBVL, 
                    USA_GPAY, USA_HI, USA_IMPX, USA_IMVOL, USA_IP, USA_IPMOM, USA_CPIC, USA_CPICM, 
                    USA_JBO, USA_LFPR, USA_LMCI, USA_LEI, USA_MPAY, USA_MP, USA_NAHB, USA_NLTTF, 
                    USA_NFIB, USA_NFP, USA_NMPMI, USA_NPP, USA_EMPST, USA_PHS, USA_PFED, USA_PP, 
                    USA_PPIC, USA_RSM, USA_RSY, USA_RSEA, USA_RFMI, USA_TVS, USA_UNR, USA_WINV, 
                    exposure, equity, settings):

    # DATE: numpy array shape = (settings['lookback'], ). Each element is an interger instead of string
    #       Example: 1991124 => 1991-12-24
    # OPEN: numpy array shape = (settings['lookback'], len(setting['markets']))
    # USA_BC: numpy array shape = (settings[lookback'], 1)
    #       If USA_BC didn't have a value at a certain date, its corresponding value is nan. 
    # When evaluating, myTradingSytem will be called for all time windows. 
    #   A time window = [starting date: starting_date + lookback]
    #   starting date iterates in [19910101: Now - lookback]
    
    
    #print '--------------'
    #print DATE.shape
    #print OPEN.shape
    #print USA_BC.shape
    #print DATE[:10]
    #print DATE[-10:]
    #print USA_DFMI[:10]
    #print USA_DFMI[-10:]

    nMarkets=CLOSE.shape[1]
    pos=numpy.ones((1,nMarkets))
    pos=pos/numpy.sum(abs(pos))

    return pos, settings



##### Do not change this function definition #####
def mySettings():
    '''Define your market list and other settings here.

    The function name "mySettings" should not be changed.

    Default settings are shown below.'''

    # Default competition and evaluation mySettings
    settings= {}

    futuresList = ['F_AD', 'F_BO', 'F_BP', 'F_C', 'F_CC', 'F_CD', 'F_CL', 'F_CT', 'F_DX', 'F_EC', 'F_ED', 'F_ES', 'F_FC', 'F_FV', 'F_GC', 'F_HG', 'F_HO', 'F_JY', 'F_KC', 'F_LB', 'F_LC', 'F_LN', 'F_MD', 'F_MP', 'F_NG', 'F_NQ', 'F_NR', 'F_O', 'F_OJ', 'F_PA', 'F_PL', 'F_RB', 'F_RU', 'F_S', 'F_SB', 'F_SF', 'F_SI', 'F_SM', 'F_TU', 'F_TY', 'F_US', 'F_W', 'F_XX', 'F_YM', 'F_AX', 'F_CA', 'F_DT', 'F_UB', 'F_UZ']
    stocksList = ['AAPL', 'MMM', 'ABT', 'ABBV', 'ACN', 'ALL', 'MO', 'AMZN', 'AEP', 'AXP', 'AIG', 'AMGN', 'APC', 'APA', 'AAPL', 'T', 'BAC', 'BK', 'BAX', 'BRK.B', 'BA', 'BMY', 'COF', 'CAT', 'CVX', 'CSCO', 'C', 'KO', 'CL', 'CMCSA', 'COP', 'COST', 'CVS', 'DVN', 'DOW', 'DD', 'EBAY', 'EMC', 'EMR', 'EXC', 'XOM', 'FB', 'FDX', 'F', 'FCX', 'GD', 'GE', 'GM', 'GILD', 'GS', 'GOOG', 'GOOGL', 'HAL', 'HPQ', 'HD', 'HON', 'INTC', 'IBM', 'JNJ', 'JPM', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDT', 'MRK', 'MET', 'MSFT', 'MDLZ', 'MON', 'MS', 'NOV', 'NKE', 'NSC', 'OXY', 'ORCL', 'PEP', 'PFE', 'PM', 'PG', 'QCOM', 'RTN', 'SLB', 'SPG', 'SO', 'SBUX', 'TGT', 'TXN', 'TWX', 'FOXA', 'USB', 'UNP', 'UPS', 'UTX', 'UNH', 'VZ', 'V', 'WBA', 'WMT', 'DIS', 'WFC', 'A', 'AA', 'ABC', 'ACE', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'ADT', 'AEE', 'AEP', 'AES', 'AET', 'AFL', 'AGN', 'AIV', 'AIZ', 'AKAM', 'ALLE', 'ALTR', 'ALXN', 'AMAT', 'AME', 'AMG', 'AMP', 'AMT', 'AN', 'AON', 'APD', 'APH', 'ARG', 'ATI', 'AVB', 'AVGO', 'AVP', 'AVY', 'AZO', 'BBBY', 'BBT', 'BBY', 'BCR', 'BDX', 'BEN', 'BF.B', 'BHI', 'BLK', 'BLL', 'BMS', 'BRCM', 'BSX', 'BWA', 'BXP', 'CA', 'CAG', 'CAH', 'CAM', 'CB', 'CBG', 'CBS', 'CCE', 'CCI', 'CCL', 'CELG', 'CERN', 'CF', 'CHK', 'CHRW', 'CI', 'CINF', 'CLX', 'CMA', 'CME', 'CMG', 'CMI', 'CMS', 'CNP', 'CNX', 'COG', 'COH', 'COL', 'CPB', 'CRM', 'CSC', 'CSX', 'CTAS', 'CTL', 'CTSH', 'CTXS', 'CVC', 'D', 'DAL', 'DE', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DISCA', 'DISCK', 'DLPH', 'DLTR', 'DNB', 'DNR', 'DO', 'DOV', 'DPS', 'DRI', 'DTE', 'DUK', 'DVA', 'EA', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EOG', 'EQR', 'EQT', 'ESRX', 'ESS', 'ESV', 'ETFC', 'ETN', 'ETR', 'EW', 'EXPD', 'EXPE', 'FAST', 'FE', 'FFIV', 'FIS', 'FISV', 'FITB', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOSL', 'FSLR', 'FTI', 'FTR', 'GAS', 'GCI', 'GGP', 'GIS', 'GLW', 'GMCR', 'GME', 'GNW', 'GPC', 'GPS', 'GRMN', 'GT', 'GWW', 'HAR', 'HAS', 'HBAN', 'HCBK', 'HCN', 'HCP', 'HES', 'HIG', 'HOG', 'HOT', 'HP', 'HRB', 'HRL', 'HRS', 'HST', 'HSY', 'HUM', 'ICE', 'IFF', 'INTU', 'IP', 'IPG', 'IR', 'IRM', 'ISRG', 'ITW', 'IVZ', 'JCI', 'JEC', 'JNPR', 'JOY', 'JWN', 'K', 'KEY', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KORS', 'KR', 'KSS', 'KSU', 'L', 'LB', 'LEG', 'LEN', 'LH', 'LLL', 'LLTC', 'LM', 'LNC', 'LRCX', 'LUK', 'LUV', 'LVLT', 'LYB', 'M', 'MAC', 'MAR', 'MAS', 'MAT', 'MCHP', 'MCK', 'MCO', 'MHFI', 'MHK', 'MJN', 'MKC', 'MLM', 'MMC', 'MNK', 'MNST', 'MOS', 'MPC', 'MRO', 'MSI', 'MTB', 'MU', 'MUR', 'MYL', 'NAVI', 'NBL', 'NBR', 'NDAQ', 'NE', 'NEE', 'NEM', 'NFLX', 'NFX', 'NI', 'NLSN', 'NOC', 'NRG', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWSA', 'OI', 'OKE', 'OMC', 'ORLY', 'PAYX', 'PBCT', 'PBI', 'PCAR', 'PCG', 'PCL', 'PCLN', 'PCP', 'PDCO', 'PEG', 'PFG', 'PGR', 'PH', 'PHM', 'PKI', 'PLD', 'PNC', 'PNR', 'PNW', 'POM', 'PPG', 'PPL', 'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PX', 'PXD', 'QEP', 'R', 'RAI', 'REGN', 'RF', 'RHI', 'RHT', 'RIG', 'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'SCG', 'SCHW', 'SE', 'SEE', 'SHW', 'SJM', 'SNA', 'SNDK', 'SNI', 'SPLS', 'SRCL', 'SRE', 'STI', 'STJ', 'STT', 'STX', 'STZ', 'SWK', 'SWN', 'SYK', 'SYMC', 'SYY', 'TAP', 'TDC', 'TE', 'TEL', 'THC', 'TIF', 'TJX', 'TMK', 'TMO', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSO', 'TSS', 'TWC', 'TXT', 'TYC', 'UA', 'UHS', 'UNM', 'URBN', 'URI', 'VAR', 'VFC', 'VIAB', 'VLO', 'VMC', 'VNO', 'VRSN', 'VRTX', 'VTR', 'WAT', 'WDC', 'WEC', 'WFM', 'WHR', 'WIN', 'WM', 'WMB', 'WU', 'WY', 'WYN', 'WYNN', 'XEC', 'XEL', 'XL', 'XLNX', 'XRAY', 'XRX', 'XYL', 'YHOO', 'YUM', 'ZION', 'ZTS' ]

    # Futures Contracts
#    settings['markets']  = ['CASH'] + futuresList + stocksList
    settings['markets']  = ['CASH'] + futuresList + stocksList

    settings['lookback']= 504  # This doesn't affect downloading data. Data download will download the full time frame. 
    settings['budget']= 10**6
    settings['slippage']= 0.05
    settings['participation']= 0.1

    return settings

# Evaluate trading system defined in current file.
if __name__ == '__main__':
    import quantiacsToolbox
    results = quantiacsToolbox.runts(__file__)
