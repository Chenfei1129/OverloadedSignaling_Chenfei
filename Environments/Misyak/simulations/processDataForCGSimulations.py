import pandas as pd


def main():
	cgSim = pd.read_pickle('/home/stacyste/Documents/Research/OverloadedSignaling/Environments/Misyak/data/simulations/conditionsForCommonGround2020-8-25.pkl')
	cgSim.reset_index(level=0, inplace=True)
	cgSim['axesInCG'] = 0 
	cgSim['tokensInCG'] = 0 
	cg = pd.concat([cgSim, cgSim, cgSim, cgSim], ignore_index=True)
	cg.loc[0:2047,'axesInCG'] = 1
	cg.loc[0:1023,'tokensInCG'] = 1
	cg.loc[2048:3072,'tokensInCG'] = 1
	cg.to_pickle('../commonGroundExptTrials.pkl')

if __name__ == "__main__":
	main()