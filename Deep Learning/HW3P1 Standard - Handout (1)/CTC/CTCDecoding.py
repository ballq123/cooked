import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        prev = -1
        T = y_probs.shape[0]
        for t in range(T):
            bestProb = 0
            for y in range(len(y_probs)):
                curProb = y_probs[y][t]
                if curProb > bestProb:
                    maxP = curProb
                    maxIdx = y
            path_prob = path_prob * maxP
            symb = self.symbol_set[maxIdx - 1]
            if (prev != symb) and (maxIdx != blank):
                decoded_path.append(symb)
            prev = self.symbol_set[maxIdx - 1]

        decoded_path = ''.join(decoded_path)

        return decoded_path, path_prob
        # raise NotImplementedError


class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """

        T = y_probs.shape[1]
        beam_blank = [""]  
        score_blank = {"": y_probs[0, 0, 0]}
        beam_symbol = [s for s in self.symbol_set]
        score_symbol = {s: y_probs[i + 1, 0, 0] for i, s in enumerate(self.symbol_set)}

        for t in range(T):
            if t > 0:
                beam_blank, score_blank, beam_symbol, score_symbol = self.prune(
                    beam_blank, score_blank, beam_symbol, score_symbol
                )
                new_beam_blank, new_score_blank = self.extend_with_blanks(
                    y_probs, beam_blank, score_blank, beam_symbol, score_symbol, t
                )
                new_beam_symbol, new_score_symbol = self.extend_with_symbols(
                    y_probs, beam_blank, score_blank, beam_symbol, score_symbol, t
                )
                beam_blank, score_blank = new_beam_blank, new_score_blank
                beam_symbol, score_symbol = new_beam_symbol, new_score_symbol

        best_path, final_scores = self.merge_paths(beam_blank, score_blank, beam_symbol, score_symbol)
        return best_path, final_scores

    def prune(self, beam_blank, score_blank, beam_symbol, score_symbol):
        all_scores = list(score_blank.values()) + list(score_symbol.values())
        sorted_scores = sorted(all_scores)
        cutoff = sorted_scores[-self.beam_width] if len(sorted_scores) >= self.beam_width else min(sorted_scores)
        pruned_blank = [p for p in beam_blank if score_blank[p] >= cutoff]
        pruned_blank_scores = {p: score_blank[p] for p in pruned_blank}
        pruned_symbol = [p for p in beam_symbol if score_symbol[p] >= cutoff]
        pruned_symbol_scores = {p: score_symbol[p] for p in pruned_symbol}
        return pruned_blank, pruned_blank_scores, pruned_symbol, pruned_symbol_scores

    def extend_with_blanks(self, y_probs, beam_blank, score_blank, beam_symbol, score_symbol, t):
        ext_blank = {}
        for path in beam_blank:
            ext_blank[path] = score_blank[path] * y_probs[0, t, 0]
        for path in beam_symbol:
            ext_blank[path] = ext_blank.get(path, 0) + score_symbol[path] * y_probs[0, t, 0]
        return list(ext_blank.keys()), ext_blank

    def extend_with_symbols(self, y_probs, beam_blank, score_blank, beam_symbol, score_symbol, t):
        ext_symbol = {}
        new_paths = []
        for path in beam_blank:
            for i, sym in enumerate(self.symbol_set):
                new_path = path + sym
                ext_symbol[new_path] = ext_symbol.get(new_path, 0) + score_blank[path] * y_probs[i + 1, t, 0]
                if new_path not in new_paths:
                    new_paths.append(new_path)
        for path in beam_symbol:
            for i, sym in enumerate(self.symbol_set):
                # dont repeat the same symbol
                new_path = path if path and path[-1] == sym else path + sym
                ext_symbol[new_path] = ext_symbol.get(new_path, 0) + score_symbol[path] * y_probs[i + 1, t, 0]
                if new_path not in new_paths:
                    new_paths.append(new_path)
        return new_paths, ext_symbol

    def merge_paths(self, beam_blank, score_blank, beam_symbol, score_symbol):
        merged_scores = score_blank.copy()
        for path in beam_symbol:
            merged_scores[path] = merged_scores.get(path, 0) + score_symbol[path]
        best = max(merged_scores, key=merged_scores.get)
        return best, merged_scores
        # raise NotImplementedError
