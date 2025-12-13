import src.tokens as tokens
from src.sequence.parsing_errors import FloorPlanSequenceParsingError
from src.floor_plan_tokenizer import FloorPlanTokenizer

import shapely

from src.sequence import from_sequence

from src.geom_utils import TurnType, turn_type

import torch
import torch.nn.functional as F


def gaussian1d(mu,sigma,res,device='cpu'):
    """
    Creates a discrete non-normalized gaussian PDF in the range [0,res-1]

    Parameters:
    mu (float): center of the normal distribution
    sigma (float): variance of the normal distribution
    res (int): determines the range [0,res-1] in which the normal distribution is evaluated
    device (String): device of the returned tensor
    Returns:
    1D-tensor (float): tensor with shape [res] containing the evaluated PDF
    """
    mu = torch.as_tensor(mu).view(-1,1)
    x = torch.linspace(0, res-1, res,device=device).view(1,-1)
    return torch.exp(-0.5*((x-mu)/sigma)**2) #* 1/(s*np.sqrt(2*np.pi)) 


class NarrowSpacesLoss:
    res = 256

    def __init__(self, base_loss, tokenizer: FloorPlanTokenizer, alpha=10.0):
        self.base_loss = base_loss
        self.alpha = alpha
        self.tokenizer: FloorPlanTokenizer = tokenizer

    def __call__(self, output, labels: torch.Tensor):
        std_loss = self.base_loss(output, labels)

        logits = output.logits

        coords_mask = self.tokenizer.is_coord_token(labels)
        prompt_mask = self.tokenizer.prompt_mask(labels)

        added_coords_tokens = coords_mask & (~prompt_mask)
        
        for i in range(labels.size(0)):
            interp_val = self._get_predictions(logits[i,:,:], labels[i,:])


        return std_loss


    def _get_predictions(self, logits, labels, added_tokens_mask):
        device = labels.device

        

        probs = F.softmax(logits,-1)
        pred_ind = torch.argmax(probs,-1)

        if not self.tokenizer.is_coord_token(pred_ind):
            return torch.nan

        gauss_weights = gaussian1d(pred_ind,sigma=1.0,res=self.res,device=device)

        row_ind = torch.linspace(0,logits.size(0)-1,logits.size(0),dtype=torch.long,device=device)
        col_ind = torch.linspace(0,logits.size(1)-2,logits.size(1)-1,dtype=torch.long,device=device)

        interp_val = torch.sum(col_ind * probs[row_ind,:-1] * gauss_weights,1)
        prob_sum = torch.sum(probs[row_ind,:-1] * gauss_weights,1)
        interp_val[prob_sum > 0.0] = interp_val[prob_sum > 0.0] / prob_sum[prob_sum > 0.0]
        interp_val = torch.cat([torch.zeros(1,device=device),interp_val[:-2]]).view(-1,6)
        label_mat = labels[:-1].view(-1,6).clone()
        return interp_val, label_mat


    def evaluate(self, seq: torch.Tensor):
        decoded = self.tokenizer.decode(seq)
        
        try:
            plan = from_sequence(decoded)
        except FloorPlanSequenceParsingError:
            return 5.0
        
        loss = torch.tensor(0)

        for room in plan.rooms:
            room_coords = room.polygon().exterior.coords
            room_coords = room_coords[:-1]                  # Remove duplicated last point

            for i in range(len(room_coords)):
                p0 = room_coords[i]
                p1 = room_coords[i-1]

                segment = shapely.LineString([p0, p1])
                

                loss += 1.0 / (self.alpha * segment.length)

        
                
