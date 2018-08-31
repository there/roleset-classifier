import glob
import pandas as pd
import os
import tensorflow as tf
import tensorflow_hub as hub
import re
import shutil

def load_data(filename):
    n_rolesets = 1
    
    data = {}
    data['prev_word'] = []
    data['target_word'] = []
    data['next_word'] = []
    
    data['prev_pos'] = []
    data['target_pos'] = []
    data['next_pos'] = []
    
    data['prev_sr'] = []
    data['target_sr'] = []
    data['next_sr'] = []
    
    data['prev_lexname'] = []
    data['next_lexname'] = []
    
    data['roleset'] = []
    with open(filename, 'r') as f:
            prog = re.compile(r'^(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t([\S ]+)\t([\S ]+)\t([\S ]+)\t([\S ]+)\t([\S ]+)\t([\S ]+)\t(\d+)$')
            for line in f:
                result = prog.match(line)
                if result:
                    #data['ngram'].append(result.group(1))
                    data['prev_word'].append(result.group(1))
                    data['target_word'].append(result.group(2))
                    data['next_word'].append(result.group(3))
                    
                    data['prev_pos'].append(result.group(4))
                    data['target_pos'].append(result.group(5))
                    data['next_pos'].append(result.group(6))
                    data['prev_sr'].append(result.group(7))
                    data['target_sr'].append(result.group(8))
                    data['next_sr'].append(result.group(9))
                    
                    data['prev_lexname'].append(result.group(14))
                    data['next_lexname'].append(result.group(15))
                    
                    roleset = int(result.group(16))
                    if roleset > n_rolesets: n_rolesets = roleset
                    
                    data['roleset'].append(roleset)
    return pd.DataFrame.from_dict(data), n_rolesets

pos_list = ['START', 'END', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'HYPH', '\'\'', '.', ':', ',', ')', '(', '``', '$' ]
sr_list = ['START', 'END', 'ROOT', 'neg', 'acl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'case', 'cc', 'ccomp', 'clf', 'compound', 'conj', 'cop', 'csubj', 'dep', 'det', 'discourse', 'dislocated', 'expl', 'fixed', 'flat', 'goeswith', 'iobj', 'list', 'mark', 'nmod', 'nsubj', 'nummod', 'obj', 'obl', 'orphan', 'parataxis', 'punct', 'reparandum', 'root', 'vocative', 'xcomp']
lexname_list = ['START', 'END', ' ', 'adj.all', 'adj.pert', 'adv.all', 'noun.Tops', 'noun.act', 'noun.animal', 'noun.artifact', 'noun.attribute', 'noun.body', 'noun.cognition', 'noun.communication', 'noun.event', 'noun.feeling', 'noun.food', 'noun.group', 'noun.location', 'noun.motive', 'noun.object', 'noun.person', 'noun.phenomenon', 'noun.plant', 'noun.possession', 'noun.process', 'noun.quantity', 'noun.relation', 'noun.shape', 'noun.state', 'noun.substance', 'noun.time', 'verb.body', 'verb.change', 'verb.cognition', 'verb.communication', 'verb.competition', 'verb.consumption', 'verb.contact', 'verb.creation', 'verb.emotion', 'verb.motion', 'verb.perception', 'verb.possession', 'verb.social', 'verb.stative', 'verb.weather', 'adj.ppl']

class ExampleCheckpointSaverListener(tf.train.CheckpointSaverListener):
    def __init__(self, estimator, dev_input_fn, best_accuracy):
        self.estimator = estimator
        self.dev_input_fn = dev_input_fn
        self.best_accuracy = best_accuracy
        
    def begin(self):
        # You can add ops to the graph here.
        print('Starting the session.')

    def before_save(self, session, global_step_value):
        results = self.estimator.evaluate(input_fn=self.dev_input_fn)
        print(results)
        print('Best accuracy: ', self.best_accuracy)
        if results['accuracy'] >= self.best_accuracy:
            self.best_accuracy = results['accuracy']

    def after_save(self, session, global_step_value):
        print('Checkpoint saved')

    def end(self, session, global_step_value):
        print('Done with the session.')

def create_shared_embedding_columns(keys, vocabulary_list, dimension):
    column_list = []
    for key in keys:
        column_list.append(
            tf.feature_column.categorical_column_with_vocabulary_list(
                key=key,
                vocabulary_list=vocabulary_list))
    return tf.feature_column.shared_embedding_columns(column_list, dimension=dimension)

def train_and_evaluate_with_module(hub_module, lemma, train_module=False):
    train_df, n_rolesets = load_data('processed/' + lemma + '_train_combine_examples_unaugmented.txt')
    dev_df, dev_rolesets = load_data('processed/' + lemma + '_dev_combine_examples.txt')
    test_df, test_rolesets = load_data('processed/' + lemma + '_test_combine_examples.txt')    

    if dev_rolesets > n_rolesets: n_rolesets = dev_rolesets
    if test_rolesets > n_rolesets: n_rolesets = test_rolesets    

    # training input on whole training set with no limit on training epochs
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df['roleset'], num_epochs=None, shuffle=True)

    # dev input on whole training set with no limit on training epochs
    dev_input_fn = tf.estimator.inputs.pandas_input_fn(
    dev_df, dev_df['roleset'], num_epochs=None, shuffle=False)

    # prediction on whole training set
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df['roleset'], batch_size=32, shuffle=True)

    # training input on whole dev set
    predict_dev_input_fn = tf.estimator.inputs.pandas_input_fn(
    dev_df, dev_df['roleset'], batch_size=32, shuffle=False)

    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df['roleset'], batch_size=32, shuffle=False)

    #shared embeddiing columns
    pos_feature_columns = create_shared_embedding_columns(['prev_pos', 'target_pos', 'next_pos'], pos_list, 5)
    sr_feature_columns = create_shared_embedding_columns(['prev_sr', 'target_sr', 'next_sr'], sr_list, 12)
    lexname_feature_columns = create_shared_embedding_columns(['prev_lexname', 'next_lexname'], lexname_list, 4)

    text_feature_columns = [
        hub.text_embedding_column(key='prev_word', module_spec=hub_module, trainable=train_module),
        hub.text_embedding_column(key='target_word', module_spec=hub_module, trainable=train_module),
        hub.text_embedding_column(key='next_word', module_spec=hub_module, trainable=train_module)]

    model_dir = "/home/tphan/Desktop/python/2018_summer/classifier/model/" + lemma
    feature_columns = text_feature_columns+pos_feature_columns+sr_feature_columns+lexname_feature_columns
    #estimator = tf.estimator.BaselineClassifier(n_classes=n_rolesets+1,model_dir=model_dir)
    estimator = tf.estimator.DNNClassifier(
        config=tf.estimator.RunConfig(keep_checkpoint_max=16),
        hidden_units=[500,100],
        feature_columns=text_feature_columns+pos_feature_columns+sr_feature_columns+lexname_feature_columns,
        n_classes=n_rolesets+1,
        optimizer=tf.train.AdamOptimizer(),
        dropout=.5,
        model_dir=model_dir)
    os.makedirs(estimator.eval_dir())

    best_accuracy = -1
    listener = ExampleCheckpointSaverListener(estimator, predict_dev_input_fn, best_accuracy)
    saver_hook = tf.train.CheckpointSaverHook(model_dir, listeners=[listener], save_steps=2000)
    hook = tf.contrib.estimator.stop_if_no_decrease_hook(estimator, 'loss', 8000, min_steps=8000, run_every_secs=None, run_every_steps=2000)
    estimator.train(input_fn=train_input_fn, hooks=[hook, saver_hook])
    checkpoint_state_proto = tf.train.get_checkpoint_state(model_dir)
    best_results = {}
    best_predictions = []
    best_checkpoint_path = None
    test_results = {}
    test_predictions = []
    if checkpoint_state_proto is not None:
        checkpoint_paths = checkpoint_state_proto.all_model_checkpoint_paths
        if len(checkpoint_paths) > 0:
            best_accuracy = 0
            for checkpoint_path in checkpoint_paths:
                results = estimator.evaluate(input_fn=predict_dev_input_fn, checkpoint_path=checkpoint_path)
                if results['accuracy'] >= best_accuracy:
                    best_checkpoint_path = checkpoint_path
                    best_accuracy = results['accuracy']
                    best_results = results

            for pred_dict in estimator.predict(predict_dev_input_fn, checkpoint_path=best_checkpoint_path):
                best_predictions.append(pred_dict)
            for pred_dict in estimator.predict(predict_test_input_fn, checkpoint_path=best_checkpoint_path):
                test_predictions.append(pred_dict)
            
            test_results = estimator.evaluate(input_fn=predict_test_input_fn, checkpoint_path=best_checkpoint_path)
            print('Results')
            print(best_results)
            print('Test results')
            print(test_results)
            #estimator.export_savedmodel('/home/tphan/Desktop/python/2018_summer/classifier/savedmodel/' + lemma,
            #                            tf.estimator.export.build_parsing_serving_input_receiver_fn(
            #                                tf.feature_column.make_parse_example_spec(feature_columns),
            #                                32),
            #                            checkpoint_path=checkpoint_paths[0],
            #                            strip_default_attrs=True)
            shutil.rmtree(model_dir)
            return best_predictions, test_predictions, {
                'Rolesets': str(int(n_rolesets)),
                'Rolesets in dev': str(int(dev_rolesets)),
                'Train size': str(int(train_df.shape[0])),
                'Dev size': str(int(dev_df.shape[0])),
                'Dev accuracy': best_results['accuracy'],
                'Average loss': best_results['average_loss'],
                'Loss': best_results['loss'],
                'Global step': best_results['global_step']
            }, {
                'Rolesets': str(int(n_rolesets)),
                'Rolesets in test': str(int(test_rolesets)),
                'Train size': str(int(train_df.shape[0])),
                'Test size': str(int(test_df.shape[0])),
                'Test accuracy': test_results['accuracy'],
                'Average loss': test_results['average_loss'],
                'Loss': test_results['loss'],
                'Global step': test_results['global_step']
            }
    return None, None

lemmata = ['be', 'have', 'say', 'do', 'get', 'go', 'make', 'use', 'take', 'know', 'see', 'come', 'call', 'work', 'add', 'find', 'help', 'pay', 'try', 'look', 'operate', 'hold', 'tell', 'continue', 'serve', 'show', 'become', 'end', 'move', 'fall', 'concern', 'put', 'ask', 'keep', 'leave', 'change', 'run', 'spend', 'raise', 'lead', 'start', 'meet', 'feel', 'consider', 'send', 'develop', 'charge', 'build', 'lose', 'close', 'allow', 'grow', 'talk', 'mean', 'return', 'order', 'decline', 'cut', 'note', 'file', 'name', 'deal', 'base', 'act', 'live', 'appear', 'reach', 'view', 'follow', 'open', 'drop', 'manage', 'play', 'set', 'succeed', 'stop', 'happen', 'vote', 'turn', 'yield', 'stay', 'improve', 'question', 'pass', 'finance', 'drive', 'force', 'cover', 'stand', 'settle', 'argue', 'claim', 'fly', 'apply', 'process', 'review', 'rule', 'assume', 'love', 'introduce', 'resign', 'hit', 'affect', 'push', 'join', 'bid', 'sign', 'head', 'watch', 'treat', 'enter', 'care', 'gain', 'confer', 'refer', 'back', 'strike', 'walk', 'prepare', 'form', 'worry', 'save', 'break', 'bear', 'cite', 'compete', 'admit', 'encourage', 'aim', 'list', 'perform', 'measure', 'conclude', 'hurt', 'fill', 'ease', 'retire', 'contend', 'recover', 'realize', 'sound', 'slow', 'jump', 'draw', 'trip', 'fix', 'conduct', 'execute', 'miss', 'identify', 'recall', 'pull', 'promote', 'approach', 'tend', 'resolve', 'discount', 'throw', 'finish', 'extend', 'clear', 'matter', 'check', 'address', 'notice', 'employ', 'appreciate', 'suppose', 'restore', 'express', 'train', 'recognize', 'mark', 'commit', 'emerge', 'stem', 'feed', 'catch', 'spread', 'shoot', 'concentrate', 'climb', 'rally', 'count', 'soar', 'differ', 'cap', 'abandon', 'register', 'prompt', 'beat', 'plunge', 'track', 'plead', 'figure', 'warm', 'submit', 'press', 'fit', 'slip', 'project', 'exercise', 'divide', 'afford', 'double', 'arrest', 'seize', 'trust', 'insure', 'fire', 'sense', 'mount', 'mind', 'sustain', 'protest', 'lift', 'split', 'pick', 'locate', 'drink', 'contract', 'wave', 'struggle', 'crash', 'appeal', 'satisfy', 'pose', 'dismiss', 'cast', 'time', 'tie', 'cross', 'secure', 'assert', 'point', 'lease', 'land', 'squeeze', 'bother', 'slide', 'sleep', 'observe', 'halt', 'abuse', 'taste', 'swing', 'scare', 'relieve', 'explode', 'depress', 'credit', 'blow', 'bill', 'smoke', 'slash', 'rent', 'manipulate', 'illustrate', 'hang', 'evolve', 'contest', 'amount', 'weigh', 'prevail', 'trim', 'impress', 'dance', 'cheat', 'scramble', 'march', 'laugh', 'ring', 'burst', 'terminate', 'race', 'smell', 'paint', 'jolt', 'incorporate', 'excuse', 'cook', 'celebrate', 'tremor', 'surface', 'rest', 'top', 'tap', 'shed', 'leap', 'crack', 'assemble', 'prescribe', 'knock', 'compose', 'capitalize', 'unload', 'tape', 'scrap', 'retreat', 'freeze', 'upgrade', 'sweep', 'subscribe', 'motivate', 'fold', 'pitch', 'mistake', 'erupt', 'crowd', 'spell', 'render', 'dispose', 'correspond', 'swear', 'spin', 'seat', 'overlook', 'filter', 'dip', 'dictate', 'condition', 'classify', 'tip', 'sway', 'strain', 'screen', 'reckon', 'entitle', 'compromise', 'boom', 'bind', 'bend', 'upset', 'tear', 'sniff', 'scuttle', 'scale', 'hail', 'flash', 'curse', 'cry', 'command', 'bond', 'wrestle', 'stir', 'refinance', 'cheer', 'bleed', 'blast', 'venture', 'spare', 'pop', 'insulate', 'grind', 'galvanize', 'frame', 'flock', 'finger', 'divorce', 'code', 'circle', 'bow', 'balloon', 'stamp', 'snap', 'scratch', 'restate', 'reassert', 'rattle', 'plug', 'pile', 'pave', 'discharge', 'dawn', 'choke', 'bust', 'smash', 'lodge', 'heave', 'drool', 'cruise', 'conceive', 'bundle', 'appraise', 'wring', 'wiggle', 'weave', 'spurt', 'slam', 'skirt', 'skid', 'screech', 'ply', 'hook', 'fume', 'dispense', 'delight', 'blunder', 'accord', 'unhinge', 'stunt', 'repaint', 'recess', 'pinch', 'overbid', 'marvel', 'level', 'inaugurate', 'buzz']
zero_dev = ['accord', 'amount', 'appraise', 'assemble', 'assert', 'balloon', 'bend', 'bind', 'bleed', 'blunder', 'boom', 'bow', 'bundle', 'bust', 'capitalize', 'cheat', 'choke', 'circle', 'classify', 'clear', 'code', 'command', 'condition', 'correspond', 'crack', 'cruise', 'curse', 'dawn', 'dictate', 'dip', 'discharge', 'dispense', 'divorce', 'drool', 'employ', 'entitle', 'erupt', 'evolve', 'exercise', 'figure', 'filter', 'finger', 'flash', 'fold', 'frame', 'fume', 'galvanize', 'grind', 'heave', 'hook', 'illustrate', 'impress', 'inaugurate', 'insure', 'jolt', 'laugh', 'leap', 'lease', 'lift', 'lodge', 'manipulate', 'measure', 'motivate', 'overbid', 'overlook', 'plead', 'pop', 'prescribe', 'project', 'race', 'reassert', 'recall', 'recess', 'refinance', 'register', 'render', 'repaint', 'restate', 'rest', 'restore', 'ring', 'scale', 'scare', 'scrap', 'scratch', 'screech', 'screen', 'seize', 'shed', 'skid', 'skirt', 'slam', 'slash', 'smash', 'snap', 'sniff', 'spare', 'spell', 'split', 'spurt', 'squeeze', 'stamp', 'stem', 'strain', 'surface', 'sway', 'swear', 'terminate', 'top', 'tremor', 'trim', 'unhinge', 'upset', 'warm', 'weave', 'wiggle', 'wrestle', 'bond', 'burst', 'buzz', 'celebrate', 'cheer', 'compose', 'cross', 'crowd', 'cry', 'delight', 'dispose', 'excuse', 'feed', 'halt', 'insulate', 'knock', 'level', 'march', 'marvel', 'pitch', 'plug', 'ply', 'reckon', 'resolve', 'retire', 'scramble', 'scuttle', 'seat', 'secure', 'slide', 'smoke', 'soar', 'spin', 'submit', 'tape', 'tap', 'tear', 'tip', 'unload', 'venture', 'weigh']
lemmata = [x for x in lemmata if x not in zero_dev]

with open('results_unaugmented_dev.txt', 'w') as dev_writer:
    with open('results_unaugmented_test.txt', 'w') as test_writer:
        for lemma in lemmata:
            dev_pred, test_pred, dev_results, test_results = train_and_evaluate_with_module('modules/1', lemma)
            if dev_results is not None:
                results_list = [lemma, str(dev_results['Dev accuracy']), str(dev_results['Average loss']), str(dev_results['Loss']), str(dev_results['Global step'])]
                dev_writer.write('\t'.join(results_list) + '\n')
            if test_results is not None:
                results_list = [lemma, str(test_results['Test accuracy']), str(test_results['Average loss']), str(test_results['Loss']), str(test_results['Global step'])]
                test_writer.write('\t'.join(results_list) + '\n')
            with open('predictions/unaugmented_' + lemma + '_mind_dev.txt', 'w') as pred_file_dev:
                for pred in dev_pred:
                    class_id = pred['class_ids'][0]
                    pred_file_dev.write('\t'.join([str(class_id), str(pred['probabilities'][class_id] * 100)]) + '\n')
            with open('predictions/unaugmented_' + lemma + '_test.txt', 'w') as pred_file_test:
                for pred in test_pred:
                    class_id = pred['class_ids'][0]
                    pred_file_test.write('\t'.join([str(class_id), str(pred['probabilities'][class_id] * 100)]) + '\n')
