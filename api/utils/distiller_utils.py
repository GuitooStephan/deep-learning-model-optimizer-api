from tensorflow import keras
import tensorflow as tf


class Distiller(keras.Model):
  def __init__(self, student, teacher):
    super(Distiller, self).__init__()
    self.teacher = teacher
    self.student = student

  def compile(
          self,
          optimizer,
          metrics,
          student_loss_fn,
          distillation_loss_fn,
          alpha=0.1,
          temperature=3,
  ):

    super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
    self.student_loss_fn=student_loss_fn
    self.distillation_loss_fn= distillation_loss_fn
    self.temperature= temperature
    self.alpha= alpha
        
  def train_step(self, data):
    x,y=data
      
    # Forward pass of teacher
    teacher_prediction=self.teacher(x, training=False)
    print("Tecaher prediction   ...", teacher_prediction)
    with tf.GradientTape() as tape:
      
      # Forward pass of student
      student_predcition= self.student(x, training=True)
      # Compute losses
      student_loss= self.student_loss_fn(y, student_predcition)
      
      distillation_loss=self.distillation_loss_fn(
        tf.nn.softmax(teacher_prediction/self.temperature, axis=1),
        tf.nn.softmax(student_predcition/self.temperature, axis=1)
      )
      loss= self.alpha* student_loss + (1-self.alpha)* distillation_loss
      print("Loss in distiller :",loss)
      # Compute gradients
      trainable_vars= self.student.trainable_variables
      gradients=tape.gradient(loss, trainable_vars)
      gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
      # Update weights
      self.optimizer.apply_gradients(zip(gradients, trainable_vars))
      
      # Update the metrics configured in `compile()`
      self.compiled_metrics.update_state(y, student_predcition)
      
      # Return a dict of performance
      results={ m.name: m.result()  for m in self.metrics}
      results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
      print("Train...", results)
      return results
        
  def test_step(self, data):
    # Unpack the data
    x, y = data
    
    ## Compute predictions
    y_prediction= self.student(x, training=False)
    
    # calculate the loss
    student_loss= self.student_loss_fn(y, y_prediction)
    
    # Update the metrics.
    self.compiled_metrics.update_state(y, y_prediction)
    
    # Return a dict of performance
    results ={m.name: m.result() for m in self.metrics}
    results.update({"student_loss": student_loss})
    print("Test...", results)
    return results

