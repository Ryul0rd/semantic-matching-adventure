import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import random


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
            'wander': self.wander,
            'go into woods': self.wander,
            'leave': self.wander,
        }
    
    def go_north(self, world):
        print('You get the feeling you shouldn\'t actually go that way.')

    def go_south(self, world):
        print('You get the feeling you shouldn\'t actually go that way.')

    def go_east(self, world):
        world.locations['goblin camp'].arrive(world)

    def go_west(self, world):
        world.locations['cabin'].arrive(world)

    def wander(self, world):
        go_x = random.choice([self.go_north, self.go_south, self.go_west, self.go_east])
        go_x(world)


class Cabin(Location):
    def __init__(self):
        super().__init__(name='cabin')
        self.first_arrival_text = 'You wander through the woods until you stumble upon a small cabin.'
        self.arrival_text = 'You return to the cabin.'
        self.actions = {
            'go north': self.go_north,
            'go south': self.go_south,
            'go east': self.go_east,
            'go west': self.go_west,
            'look in window': self.look_around,
            'look around': self.look_around,
            'knock on door': self.knock,
            'go inside': self.enter,
            'enter': self.enter,
            'break in': self.enter,
            'read a book': self.read_book,
            'leave': self.go_east,
            'go back': self.go_east,
            'take the axe': self.take_axe,
            'take the sword': self.take_sword,
        }
        self.axe_taken = False
        self.sword_taken = False
        self.gone_inside = False

    def go_north(self, world):
        print('You get the feeling you shouldn\'t actually go that way.')

    def go_south(self, world):
        print('You get the feeling you shouldn\'t actually go that way.')

    def go_east(self, world):
        world.locations['clearing'].arrive(world)

    def go_west(self, world):
        print('You get the feeling you shouldn\'t actually go that way.')

    def look_around(self, world):
        if self.gone_inside:
            print('You look around the cabin and find it modestly furnished, including a bookshelf full of books. There is no food in the pantry. There is however a sword displayed prominently on the wall.')
        else:
            print('You walk around the premises and notice a chopping block with an axe stuck in it. You decide to peek in through a window and see no one inside.')

    def knock(self, world):
        print('You walk up to the door and knock. You wait a moment but no one answers and you hear no one inside.')

    def take_axe(self, world):
        if self.axe_taken:
            print('You already took the axe here.')
        else:
            print('You take the axe from a chopping block just outside the cabin.')
            self.axe_taken = True
            world.inventory.add('axe')

    def take_sword(self, world):
        if self.gone_inside:
            if self.sword_taken:
                print('You already took the sword here.')
            else:
                print('You take the sword from its display on the wall.')
                self.sword_taken = True
                world.inventory.add('sword')
        else:
            print('You don\'t see a sword anywhere.')

    def enter(self, world):
        if self.gone_inside:
            print('You\'ve already gone inside.')
        else:
            print('You try to open the door and find it unlocked. You push the door open and cautiously enter but find no one home.')
            self.gone_inside = True

    def read_book(self, world):
        if self.gone_inside:
            print('You go to read a book from the book shelf but find that all the books are in a language you don\'t know.')
        else:
            print('You don\'t see a book anywhere.')


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
        self.match_threshold = 0.45
        self.inventory = set()
        self.locations = {}
        self.locations.update(name_and_location(Clearing()))
        self.locations.update(name_and_location(Cabin()))
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
