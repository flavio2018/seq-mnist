import torch
from torch import nn
import torch.nn.functional as F


class DynamicNeuralTuringMachineMemory(nn.Module):
    def __init__(self, n_locations, content_size, address_size, controller_input_size, controller_hidden_state_size):
        """Instantiate the memory.
        n_locations: number of memory locations
        content_size: size of the content part of memory locations
        address_size: size of the address part of memory locations"""
        super(DynamicNeuralTuringMachineMemory, self).__init__()

        self.register_buffer("memory_contents", torch.zeros(size=(n_locations, content_size)))
        # self.memory_contents = nn.Parameter(torch.zeros(size=(n_locations, content_size)), requires_grad=False)
        self.memory_addresses = nn.Parameter(torch.zeros(size=(n_locations, address_size)), requires_grad=True)
        self.overall_memory_size = content_size + address_size

        self.W_hat_hidden = nn.Parameter(torch.zeros(size=(n_locations, controller_hidden_state_size)))

        # query vector MLP parameters (W_k, b_k)
        self.W_query = nn.Parameter(torch.zeros(size=(n_locations, self.overall_memory_size)), requires_grad=True)
        self.b_query = nn.Parameter(torch.zeros(size=(self.overall_memory_size, 1)), requires_grad=True)

        # sharpening parameters (u_beta, b_beta)
        self.u_sharpen = nn.Parameter(torch.zeros(size=(controller_hidden_state_size, 1)), requires_grad=True)
        self.b_sharpen = nn.Parameter(torch.zeros(1), requires_grad=True)

        # LRU parameters (u_gamma, b_gamma)
        self.b_lru = nn.Parameter(torch.zeros(1))
        self.u_lru = nn.Parameter(torch.zeros(size=(controller_hidden_state_size, 1)))

        # erase parameters (generate e_t)
        self.W_erase = nn.Parameter(torch.zeros(size=(content_size, controller_hidden_state_size)))
        self.b_erase = nn.Parameter(torch.zeros(size=(content_size, 1)))

        # writing parameters (W_m, W_h, alpha)
        self.W_content_hidden = nn.Parameter(torch.zeros(size=(content_size, controller_hidden_state_size)))
        self.W_content_input = nn.Parameter(torch.zeros(size=(content_size, controller_input_size)))
        self.u_input_content_alpha = nn.Parameter(torch.zeros(size=(1, controller_input_size)))
        self.u_hidden_content_alpha = nn.Parameter(torch.zeros(size=(1, controller_hidden_state_size)))
        self.b_content_alpha = nn.Parameter(torch.zeros(1))

    def read(self, controller_hidden_state):
        self.read_weights = self._address_memory(controller_hidden_state)
        # this implements the memory NO-OP at reading phase
        return self._full_memory_view()[:-1, :].T @ self.read_weights[:-1, :]
        # TODO add in tests NO-OP

    def update(self, controller_hidden_state, controller_input):
        self.write_weights = self._address_memory(controller_hidden_state)
        erase_vector = self.W_erase @ controller_hidden_state + self.b_erase  # TODO MLP

        alpha = (self.u_input_content_alpha @ controller_input +
                 self.u_hidden_content_alpha @ controller_hidden_state + self.b_content_alpha)

        candidate_content_vector = F.relu(self.W_content_hidden @ controller_hidden_state +
                                          torch.mul(alpha, self.W_content_input @ controller_input))

        # this implements the memory NO-OP at writing phase
        self.memory_contents[:-1, :] = (self.memory_contents[:-1, :]
                                        - self.write_weights[:-1, :] @ erase_vector.T
                                        + self.write_weights[:-1, :] @ candidate_content_vector.T)

    def _address_memory(self, controller_hidden_state):
        projected_hidden_state = self.W_hat_hidden @ controller_hidden_state
        query = self.W_query.T @ projected_hidden_state + self.b_query
        sharpening_beta = F.softplus(self.u_sharpen.T @ controller_hidden_state + self.b_sharpen) + 1
        similarity_vector = self._compute_similarity(query, sharpening_beta)
        address_vector = self._apply_lru_addressing(similarity_vector, controller_hidden_state)
        return address_vector

    def _full_memory_view(self):
        return torch.cat((self.memory_addresses, self.memory_contents), dim=1)

    def _compute_similarity(self, query, sharpening_beta):
        """Compute the sharpened cosine similarity vector between the query and the memory locations."""
        full_memory_view = self._full_memory_view()
        return sharpening_beta * F.cosine_similarity(full_memory_view[:, None, :], query.T[None, :, :],
                                                     eps=1e-7, dim=-1)

    def _apply_lru_addressing(self, similarity_vector, controller_hidden_state):
        """Apply the Least Recently Used addressing mechanism. This shifts the addressing towards positions
        that have not been recently read or written."""
        lru_gamma = torch.sigmoid(self.u_lru.T @ controller_hidden_state + self.b_lru)
        lru_similarity_vector = F.softmax(similarity_vector - lru_gamma * self.exp_mov_avg_similarity, dim=0)
        with torch.no_grad():
            self.exp_mov_avg_similarity = 0.1 * self.exp_mov_avg_similarity + 0.9 * similarity_vector
        return lru_similarity_vector

    def _reset_memory_content(self):
        """This method exists to implement the memory reset at the beginning of each episode."""
        self.memory_contents.fill_(0)
        self.memory_contents.detach_()
        # self.memory_contents = torch.zeros_like(self.memory_contents)  # alternative

    def _reshape_and_reset_exp_mov_avg_sim(self, batch_size, device):
        with torch.no_grad():
            n_locations = self.memory_addresses.shape[0]
        self.register_buffer("exp_mov_avg_similarity", torch.zeros(size=(n_locations, batch_size)))
        self.exp_mov_avg_similarity = self.exp_mov_avg_similarity.to(device)

    # def reshape_and_reset_read_write_weights(self, shape):
    #     self.read_weights = nn.Parameter(torch.zeros(size=shape))
    #     self.write_weights = nn.Parameter(torch.zeros(size=shape))

    def forward(self, x):
        raise RuntimeError("It makes no sense to call the memory module on its own. "
                           "The module should be accessed by the controller "
                           "either by addressing, reading or updating the memory.")
