The latest version of this package can be downloaded from
https://github.com/Stezii/gSSJoin

Copyright (C) 2017 by Sidney Ribeiro Junior, Stezii

All files in this package may be modified and/or distributed according to
the GNU GPL, version 2, June 1991, that should have been distributed with
this package.

cuCompactor were developed by Davide Spataro: https://github.com/knotman90/cuStreamComp
We used some code developed by Wisllay Vitrio and Mateus Freitas: https://github.com/mateusffreitas/FT-kNN

Usage: <executable> <input_token_file> <input_weights_file> <threshold> <topk> <topk_mode> <output_file> <number_of_gpus>

<input_token_file> Each line is a record starting with its id followed by the ids of the elements contained in the set
<input_weights_file> Each line is an id followed by its integer weight
<threshold> Select the pairs with similarity greater than the threshold
<topk> For each record, select the K pairs with highest similarity
<topk_mode> In case the pairs K+1th,K+2th,... have the same similarity as the Kth pair, these can be omitted ('strict') or included ('soft')
<output_file> Each line consists a pair of ids and its similarity
<number_of_gpus> Single- or multi-GPU run