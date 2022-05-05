package ticTacToe;


import java.util.Date;
import java.util.*;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should runs/alternate (1) and (2) until convergence. 
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration: Convergence of the Values of the current policy, 
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by much (i.e. the maximum improvement is less than 
 * some small delta). The latter happens when the policy improvement step no longer updates the policy, i.e. the current policy 
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current policy (policy evaluation). 
	 */
	HashMap<Game, Double> policyValues=new HashMap<Game, Double>();
	
	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}. 
	 */
	HashMap<Game, Move> curPolicy=new HashMap<Game, Move>();
	
	double discount=0.9;
	
	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;
	
	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
		
		
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);
		
	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP paramters (rewards, transitions, etc) as specified in 
	 * {@link TTTMDP}
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		this.mdp=new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		this.mdp=new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}
	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all states to 0 
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.policyValues.put(g, 0.0);
		
	}
	
	/**
	 *  You should implement this method to initially generate a random policy, i.e. fill the {@link #curPolicy} for every state. Take care that the moves you choose
	 *  for each state ARE VALID. You can use the {@link Game#getPossibleMoves()} method to get a list of valid moves and choose 
	 *  randomly between them. 
	 */
	public void initRandomPolicy()
	{
		Random rand = new Random();
		
		for(Game game : this.policyValues.keySet()) {															
			
			List<Move> m = game.getPossibleMoves();	//storing all possible game moves in m								
			if(m.size()!= 0) 
			{															
				int number = rand.nextInt(m.size());	//random integer that acts as the index of the element							
		        Move moveList = m.get(number);			//Retrieving a random move using the random integer 								
		        curPolicy.put(game, moveList);			//storing game and move in curPolicy						
			}
			m.clear();																	
		}
	}
	
	
	/**
	 * Performs policy evaluation steps until the maximum change in values is less than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this method, 
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values of each reachable state under the current policy. 
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta)
	{
		double maxVal, sum, updateVal;
		
		//similar to value iteration, except k for loop (or fixed number of iterations)
		do {
			maxVal = 0.0;
			for (Game g : this.policyValues.keySet()) 
			{
				double currentVal = policyValues.get(g);	//assigning the current value (utility) that gets updated later on
				
				if (g.isTerminal()) 	// if the game state is terminal, then set the value function to 0.0
				{
					policyValues.put(g, 0.0);
					continue;
				}
				for (Move m : g.getPossibleMoves()) 	//for every move possible from g
				{
					sum = 0.0;
					
					//no maximising over actions
					for(TransitionProb t: mdp.generateTransitions(g,curPolicy.get(g))) 	//iterating over the current policy map
					{
						double reward = t.outcome.localReward;											
						Game prime = t.outcome.sPrime;													
						double vPrime = policyValues.get(prime);														
						
						sum += (t.prob * (reward + (discount * vPrime)));	//computing (transitionProbability * (reward + (gamma*V(s'))))
					}
					
					updateVal = Math.abs(currentVal - sum);	//calculating the updated value and using its absolute form (+ve number)
					this.policyValues.put(g,sum);	//setting the game state and policy iteration sum values

					if(updateVal > maxVal)	{
						maxVal = updateVal;	//set the V to updated value
					}
				}
			}
			
		}	while(maxVal >= delta);	//running loop until convergence has take place
	}
	
	
	/**This method should be run AFTER the {@link PolicyIterationAgent#evaluatePolicy} train method to improve the current policy according to 
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step of expectimax from each game (state) key in {@link PolicyIterationAgent#curPolicy} 
	 * to look for a move/action that potentially improves the current policy. 
	 * 
	 * @return true if the policy improved. Returns false if there was no improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy()
	{
																	
		int flag=0;	//setting flag for true and false; 0 - false
		ArrayList<Double> policyList = new ArrayList<Double>();	//list that stores policy extraction values
	
		//same as policy extraction with argmax
		for(Game g : this.curPolicy.keySet()) {		//looping over g in the current policy map														
														
			Double maxVal = 0.0;	
			Move finalMove = null;																			
	
				for(Move m : g.getPossibleMoves()) {	//for every move possible from g	
					double sum = 0.0;																		
					
					for(TransitionProb t : this.mdp.generateTransitions(g,m)) 
					{											
						sum = sum + t.prob * (t.outcome.localReward + (discount * this.policyValues.get(t.outcome.sPrime)));	//compute sum				
					}
					
					policyList.add(sum);	//add the computed sum to the list with policy extraction values																		
					maxVal = Collections.max(policyList);	//find the max value in the list
						
					//one-step expectimax 
					if(maxVal == sum) 	//if max value is equal to the sum calculated
					{          																	
						finalMove = m;	//set move as the final move that has the improved policy																		
					}
				}
				policyList.clear();																						
				
		double currentValue = policyValues.get(g);	//set current value of g																	
				
		if(maxVal > currentValue) //if policy extraction value/maxVal is greater than current value
		{																					
			curPolicy.put(g, finalMove); //then set the current policy to g and final updated move 	
			flag=1;	//return true																		
		}	
	}
		if(flag==1) 
		{
			return true;																							
		}
		else 
		{
			return false;																								
		}
	}
	
	
	/**
	 * The (convergence) delta
	 */
	double delta=0.1;
	
	/**
	 * This method should perform policy evaluation and policy improvement steps until convergence (i.e. until the policy
	 * no longer changes), and so uses your 
	 * {@link PolicyIterationAgent#evaluatePolicy} and {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train()
	{
		do 		//while the policy is still changing
		{
			this.evaluatePolicy(delta);	
		}	while (this.improvePolicy());	//till policy has been improved
		
		super.policy = new Policy(curPolicy);
	}
	
	public static void main(String[] args) throws IllegalMoveException
	{
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi=new PolicyIterationAgent();
		
		HumanAgent h=new HumanAgent();
		
		Game g=new Game(pi, h, h);
		
		g.playOut();
		
		
	}
	

}
