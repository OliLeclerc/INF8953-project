from enum import Enum

class Environment(Enum):
    ANT = "Ant-v2"
    CHEETAH = "HalfCheetah-v2"
    HOPPER = "Hopper-v2"
    SWIMMER = "Swimmer-v2"
    WALKER = "Walker2D-v2"
    HUMAN = "Humanoid-v2"

max_episode_length = 1000 # same for all for now

hp_config = {
    Environment.SWIMMER : 
    {
        "max_ep" : 100,
        "ka_penalty": 0,
        "V2": 
        {
            "n":1,
            "b":1,
            "lr":0.02,
            "std_noise": 0.01,
            "max_iter" : 1000 

        },
        "V2-t": 
        {
            "n":1,
            "b":1,
            "lr":0.02,
            "std_noise": 0.01,
            "max_iter" : 1000 

        }

    },
    
    Environment.HOPPER : 
    {
        "max_ep" : 7000,
        "ka_penalty": 1,
        "V2": 
        {
            "n":4,
            "b":4,
            "lr":0.02,
            "std_noise": 0.02,
            "max_iter" : 1000 

        },
        "V2-t": 
        {
            "n":8,
            "b":4,
            "lr":0.01,
            "std_noise": 0.025,
            "max_iter" : 1000 

        }

    },
    Environment.CHEETAH : 
    {
        "max_ep" : 12500,
        "ka_penalty": 0,
        "V2": 
        {
            "n":8,
            "b":8,
            "lr":0.02,
            "std_noise": 0.03,
            "max_iter" : 1000 

        },
        "V2-t": 
        {
            "n":32,
            "b":4,
            "lr":0.02,
            "std_noise": 0.03,
            "max_iter" : 1000 

        }

    },
    Environment.WALKER : 
    {
        "max_ep" : 75000,
        "ka_penalty": 1,
        "V2": 
        {
            "n":60,
            "b":60,
            "lr":0.025,
            "std_noise": 0.01,
            "max_iter" : 1000 

        },
        "V2-t": 
        {
            "n":40,
            "b":30,
            "lr":0.03,
            "std_noise": 0.025,
            "max_iter" : 1000 

        }

    },


    Environment.ANT : 
    {
        "max_ep" : 75000,
        "ka_penalty": 1,
        "V2": 
        {
            "n":40,
            "b":40,
            "lr":0.01,
            "std_noise": 0.025,
            "max_iter" : 1000 

        },
        "V_t": 
        {
            "n":60,
            "b":20,
            "lr":0.015,
            "std_noise": 0.025,
            "max_iter" : 1000 

        }

    },

    Environment.HUMAN : 
    {
        "max_ep" : 250000,
        "ka_penalty": 5,
        "V2": 
        {
            "n":230,
            "b":230,
            "lr":0.02,
            "std_noise": 0.0075,
            "max_iter" : 1000 

        },
        "V2-t": 
        {
            "n":230,
            "b":230,
            "lr":0.02,
            "std_noise": 0.0075,
            "max_iter" : 1000 

        }

    }
    
}