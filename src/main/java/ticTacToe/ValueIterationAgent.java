package ticTacToe;


import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to implement are: 
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free to do this, but you probably won't need to.
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction=new HashMap<Game, Double>();
	
	/**
	 * the discount factor
	 */
	double discount=0.9;
	
	/**
	 * the MDP model
	 */
	TTTMDP mdp=new TTTMDP();
	
	/**
	 * the number of iterations to perform - feel free to change this/try out different numbers of iterations
	 */
	int k=10;
	
	
	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent()
	{
		super();
		mdp=new TTTMDP();
		this.discount=0.9;
		initValues();
		train();
	}
	
	
	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);
		
	}

	public ValueIterationAgent(double discountFactor) {
		
		this.discount=discountFactor;
		mdp=new TTTMDP();
		initValues();
		train();
	}
	
	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the initial value of all states to 0 
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do this. 
	 * 
	 */
	public void initValues()
	{
		
		List<Game> allGames=Game.generateAllValidGames('X');//all valid games where it is X's turn, or it's terminal.
		for(Game g: allGames)
			this.valueFunction.put(g, 0.0);
		
		
		
	}
	
	
	
	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward, double drawReward)
	{
		this.discount=discountFactor;
		mdp=new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}
	
	/**
	 
	
	/*
	 * Performs {@link #k} value iteration steps. After running this method, the {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate()
	{
		for(int i=0; i<k; i++) 
		{
			for(Game g: this.valueFunction.keySet()) 	//updating the V(g) value of each game state
			{
				if(g.isTerminal())		// if the game state is terminal, then set the value function to 0.0
				{
					valueFunction.put(g, 0.0);
					continue;
				}
				
				double maxVal = -999999;	//highest negative v number; current maximum
				
				for(Move m: g.getPossibleMoves())	//for every move possible from g 
				{
					double sum = 0.0;
					for(TransitionProb t: mdp.generateTransitions(g,m))	//calculate Q value for every move and maximise over them
					{
						//t.outcome - t(s,a,r,s') 
						double prob = t.prob;
						double reward = t.outcome.localReward;
						double gamma = this.discount;
						double valPrime = valueFunction.get(t.outcome.sPrime);
						sum += prob * (reward + gamma*valPrime);			//computing (transitionProbability * (reward + (gamma*V(s'))))
					}		
					
					if (sum > maxVal) 
					{
						maxVal = sum;	//if the the sum is greater than the V value, update it with the new V value 
					}
				}
				this.valueFunction.put(g,maxVal);	//Updating V(g)
			}
		}
	}
	
	/**This method should be run AFTER the train method to extract a policy according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in {@link ValueIterationAgent#valueFunction} 
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy()
	{
		
		Policy policy = new Policy();	//create object of policy class 
		
		//Just like the iterate method, we calculate the game states for every move m possible and perform expectimax
		for(Game g: this.valueFunction.keySet())	//updating the V(g) value of each game state
		{
			if(g.isTerminal())		// if the game state is terminal, then set the value function to 0.0
			{
				valueFunction.put(g, 0.0);
				continue;
			}
			double maxVal = this.valueFunction.get(g);	//-999999
			
			for(Move m : g.getPossibleMoves())	//for every move possible from g
			{
				double sum = 0.0;
				for(TransitionProb t: mdp.generateTransitions(g,m))	//calculate Q value for every move and maximise over them
				{
					double prob = t.prob;
					double reward = t.outcome.localReward;
					double gamma = this.discount;
					double valPrime = valueFunction.get(t.outcome.sPrime);
					sum += prob * (reward + gamma*valPrime);		//computing (transitionProbability * (reward + (gamma*V(s'))))
				}
				
				if (sum == maxVal) 	//if the current value (sum) is the max of V, then add move as the policy
				{
					policy.policy.put(g,m);
				}
			}
		}
		
		return policy;	
	}
	
	/**
	 * This method solves the mdp using your implementation of {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}. 
	 */
	public void train()
	{
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in {@link ValueIterationAgent#valueFunction} and set the agent's policy 
		 *  
		 */
		
		super.policy=extractPolicy();
		
		if (this.policy==null)
		{
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			//System.exit(1);
		}
		
		
		
	}

	public static void main(String a[]) throws IllegalMoveException
	{
		//Test method to play the agent against a human agent.
		ValueIterationAgent agent=new ValueIterationAgent();
		HumanAgent d=new HumanAgent();
		
		Game g=new Game(agent, d, d);
		g.playOut();
		
		
		

		
		
	}
}
