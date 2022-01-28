import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class Location:
    def __init__(self, name):
        self.name = name
        self.visited = False
        self.arrival_text = f'You arrive at {self.name}.'
        self.first_arrival_text = self.arrival_text

    def arrive(self, world):
        world.current_location_name = self.name
        if self.visited:
            print(self.arrival_text)
        else:
            self.visited = True
            print(self.first_arrival_text)

class Clearing(Location):
    def __init__(self):
        super().__init__(name='clearing')
        self.first_arrival_text = '''You wake up in a small clearing in the woods. You have no memory of who you are or how you got there. You don\'t seem to have anything on you except for some simple clothes.'''
        self.arrival_text = 'You return to the clearing where you first woke up.'
        self.actions = {
            'go north': self.go_north,
            'go south': self.go_south,
            'go east': self.go_east,
            'go west': self.go_west,
            'go into woods': self.wander,
            'leave': self.wander,
        }
    
    def go_north(self, world):
        pass

    def go_south(self, world):
        pass

    def go_east(self, world):
        world.locations['goblin camp'].arrive(world)

    def go_west(self, world):
        pass

    def wander(self, world):
        pass

class GoblinCamp(Location):
    def __init__(self):
        super().__init__(name='goblin camp')
        self.actions = {
            'go west': self.go_west,
            'attack goblin': self.attack_goblin,
        }
    
    def go_west(self, world):
        world.locations['clearing'].arrive(world)

    def attack_goblin(self, world):
        print('You attack a goblin and slay it. Shortly after, the remaining goblins surround you and you are unable to excape. You try your best to fight them off but you are outnumbered and are killed.')
        print('GAME OVER')
        world.playing = False

def name_and_location(location):
    return {location.name: location}

class World:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')
        self.playing = True
        self.match_threshold = 0.6
        self.locations = {}
        self.locations.update(name_and_location(Clearing()))
        self.locations.update(name_and_location(GoblinCamp()))
        self.current_location_name = 'clearing'
        self.get_current_location().arrive(self)
    
    def get_current_location(self):
        return self.locations[self.current_location_name]

    def act(self, action):
        current_location = self.get_current_location()
        possible_actions = list(current_location.actions.keys())
        possible_action_embeddings = self.model.encode(possible_actions)
        player_action_embedding = self.model.encode(action)
        similarity_scores = cos_sim(player_action_embedding, possible_action_embeddings)
        if similarity_scores[0, torch.argmax(similarity_scores)] < self.match_threshold:
            print('Invalid Action!')
            print(possible_actions) # for debugging
            print(similarity_scores) # for debugging
        else:
            print(possible_actions[torch.argmax(similarity_scores)]) # for debugging
            print(possible_actions) # for debugging
            print(similarity_scores) # for debugging
            closest_action = possible_actions[torch.argmax(similarity_scores)]
            current_location.actions[closest_action](self)
