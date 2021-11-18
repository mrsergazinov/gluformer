import numpy as np
import pandas as pd
import pickle

class SplitData:
  def __init__(self, file_pth, ratio_list=[40,2,2] , enc_length=200, frc_length=12, scale=5):
    self.enc_length=enc_length
    self.frc_length=frc_length
    self.ratio_list=ratio_list
    self.scale=scale
    
    fileObject=open(file_pth, 'rb')
    self.all_data_wo_len_constraint, self.food_jumps = pickle.load(fileObject)
    self.full_length=self.enc_length+self.frc_length
    
    # a list of size of the people, each component is a list which contains len of connected ts 
    self.each_person_ts_list_len_wo_len_const=[list(map(len, person_i_glc)) for person_i_glc in 
                                                self.all_data_wo_len_constraint]
      
    ## Making the condition the each ts should be larger than full_len
    self.all_data_general=[]
    for i, peson_i_len_list in enumerate(self.each_person_ts_list_len_wo_len_const):
      person_i_data_w_const=[self.all_data_wo_len_constraint[i][j] for j,len_j in enumerate(peson_i_len_list)                                    if len_j>= self.full_length]
      self.all_data_general.append(person_i_data_w_const)
  
    self.each_person_ts_list_len=[list(map(len, person_i_glc)) for person_i_glc in self.all_data_general]

    # scaling the data to a proper range
    self.all_data=[]
    for i, person_i_data in enumerate(self.all_data_general):
      person_i_vl=[2*(sub_list-38)/(402-38)*self.scale-self.scale  for sub_list in person_i_data]
      self.all_data.append(person_i_vl)

    # Making three chunckes of the data based on ratio
    # list of availablities for each person 
    
    self.all_availble_train_count=[]
    self.all_cum_sum=[]
    for list_person_i_len in self.each_person_ts_list_len:
      available_spots_counts_i=[list_person_i_len_j-self.full_length+1 for list_person_i_len_j in list_person_i_len if 
                                list_person_i_len_j >= self.full_length]

      cum_sum_i=np.array(available_spots_counts_i).cumsum()
      
      self.all_availble_train_count.append(available_spots_counts_i)
      self.all_cum_sum.append(cum_sum_i)

    # total_lengthes is a list with length of number of people, with total_ength in each one
    self.total_lengthes=list(map(sum, self.all_availble_train_count))
    self.person_probs=np.array(self.total_lengthes)/sum(self.total_lengthes)
    self.num_people=len(self.person_probs)
    
    # seperating the chunks
    self.cum_ratios=np.array(ratio_list).cumsum()/sum(ratio_list)
      
  def __call__(self):
    
    # for each person
    # making init and end seeds for train, test, and valid by making proper shifiting
    train=[]
    valid=[]
    test=[]
    for i, person_i_data in enumerate(self.all_data):
      psb_length=self.total_lengthes[i]-2*self.frc_length
      
      init_seeds=[0,psb_length*self.cum_ratios[0]+self.frc_length, psb_length*self.cum_ratios[1]+2*self.frc_length]
      end_seeds=[psb_length*self.cum_ratios[0], psb_length*self.cum_ratios[1]+self.frc_length, 
                  psb_length+2*self.frc_length-1]
      
      init_indexes=[]
      end_indexes=[]
      
      for j in range(3):
        init_indexes.append(seed_to_subandSS_index(init_seeds[j], self.all_cum_sum[i]))
        end_indexes.append(seed_to_subandSS_index(end_seeds[j], self.all_cum_sum[i]))

      # making chunks by seeds and index to sets
      prs_i_train=index_to_set(person_i_data, init_indexes[0], end_indexes[0], self.full_length)
      train.append(prs_i_train)
      
      prs_i_valid=index_to_set(person_i_data, init_indexes[1], end_indexes[1], self.full_length)
      valid.append(prs_i_valid)
      
      prs_i_test=index_to_set(person_i_data, init_indexes[2], end_indexes[2], self.full_length)
      test.append(prs_i_test)
    
    
    return train, valid, test


def seed_to_subandSS_index(seed, cum_sum):
  sub_index=int(np.argmax(cum_sum>seed))
  if sub_index==0:
    sub_sub_index=int(seed)
      
  else:
    sub_sub_index=int((seed-cum_sum[sub_index-1])//1)

  return [sub_index, sub_sub_index]

def index_to_set(data, begin_indexes, end_indexes, full_length):
  output=[]
  if begin_indexes[0]== end_indexes[0]:
    output.append(data[begin_indexes[0]][begin_indexes[1]:end_indexes[1]+full_length])

      
  else:
    output.append(data[begin_indexes[0]][begin_indexes[1]:])
    if begin_indexes[0]+1==end_indexes[0]:
      output.append(data[end_indexes[0]][:end_indexes[1]+full_length])
    else:
      output+=data[begin_indexes[0]+1:end_indexes[0]]
      output.append(data[end_indexes[0]][:end_indexes[1]+full_length])
  
  return output