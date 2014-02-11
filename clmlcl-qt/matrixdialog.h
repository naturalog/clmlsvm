#ifndef MATRIXDIALOG_H
#define MATRIXDIALOG_H

#include <QDialog>
#include "defs.h"

namespace Ui {
class MatrixDialog;
}

class MatrixDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit MatrixDialog(QWidget *parent , const matrix&);
    ~MatrixDialog();
    
private:
    Ui::MatrixDialog *ui;
};

#endif // MATRIXDIALOG_H
