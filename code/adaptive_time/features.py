from abc import ABC, abstractmethod
from typing import Any, List, Dict


import adaptive_time.tiles3 as tc
import numpy as np
import itertools as iters


class Extractor(ABC):
    @abstractmethod
    def get_features(self, obs: Any) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        pass


class MountainCarTileCoder(Extractor):
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tc.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles

    def get_tiles(self, position, velocity):
        """
        Takes in a position and velocity from the mountaincar environment
        and returns a numpy array of active tiles.

        Arguments:
        position -- float, the position of the agent between -1.2 and 0.5
        velocity -- float, the velocity of the agent between -0.07 and 0.07
        returns:
        tiles - np.array, active tiles
        """
        # Set the max and min of position and velocity to scale the input
        # POSITION_MIN
        # POSITION_MAX
        # VELOCITY_MIN
        # VELOCITY_MAX
        ### START CODE HERE ###
        POSITION_MIN = -1.2
        POSITION_MAX = 0.5
        VELOCITY_MIN = -0.07
        VELOCITY_MAX = 0.07
        ### END CODE HERE ###

        # Use the ranges above and self.num_tiles to set position_scale and velocity_scale
        # position_scale = number of tiles / position range
        # velocity_scale = number of tiles / velocity range

        # Scale position and velocity by multiplying the inputs of each by their scale

        ### START CODE HERE ###
        position_scale = self.num_tiles / (POSITION_MAX - POSITION_MIN)
        velocity_scale = self.num_tiles / (VELOCITY_MAX - VELOCITY_MIN)
        ### END CODE HERE ###

        # get the tiles using tc.tiles, with self.iht, self.num_tilings and [scaled position, scaled velocity]
        # nothing to implment here
        tiles = tc.tiles(
            self.iht,
            self.num_tilings,
            [position * position_scale, velocity * velocity_scale],
        )

        return np.array(tiles)
    
    def get_features(self, obs: Any) -> np.ndarray:
        active_tiles = self.get_tiles(*obs)
        feature = np.zeros(self.iht.size)
        feature[active_tiles] = 1
        return feature

    @property
    def num_parameters(self) -> int:
        return self.iht.size


class Fourier_Features(object):
    # TODO: Implement this as an `Extractor`
    def __init__(self):
        pass
    
    def init_state_normalizers(self,maxs,mins):
        self.max = maxs
        self.min = mins
        self.range = (self.max - self.min)
    
    def normalize_state(self,state):
        return (state - self.min) / self.range
    
    def init_fourier_features(self, state_dim, order):
        self.order_list = np.array(list(iters.product(np.arange(order+1), repeat=state_dim)))

    def get_fourier_feature(self, state):
        state = self.normalize_state(state)
        order_list = self.order_list
        state_new = np.array(state).reshape(1,-1)
        scalars = np.einsum('ij, kj->ik', order_list, state_new) #do a row by row dot product with the state. i = length of order list, j = state dimensions, k = 1
        assert scalars.shape == (len(order_list),1)
        phi = np.cos(np.pi*scalars)
        return phi.flatten()


class Tabular(Extractor):
    """A simple vector feature extractor for tabular MDPs.
    
    The feature vector is a one-hot encoding of the state space. States
    are assumed to be integers from 0 to num_states-1.
    """

    def __init__(self, num_states: int):
        self._num_states = num_states

    def get_features(self, obs: Any) -> np.ndarray:
        feature = np.zeros(self._num_states)
        feature[obs] = 1
        return feature

    @property
    @abstractmethod
    def num_parameters(self) -> int:
        return self._num_states
