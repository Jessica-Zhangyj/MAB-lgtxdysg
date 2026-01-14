#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>

#define MAX_SIZE 1000000
#define EPS 1e-12
#define TRUE 1
#define FALSE 0
#define MAX_POLY_NUM 11

typedef struct{
    float coef; //系数
    int expn; //指数
}ElemType;

// Define Status as an integer type for return values
typedef int Status;

typedef struct node{
    ElemType data; //数据域
    struct node *next; //指针域
}LNode, *LinkList;

typedef LinkList polynomial; //多项式类型
typedef LinkList Position;   //定义 Position 为 LinkList 类型
polynomial P[MAX_POLY_NUM]; //定义多项式数组



// 定义基本操作函数
//初始化链表
void initList(polynomial &P){
    P = (LinkList)malloc(sizeof(LNode));
    if (!P) {
        printf("内存分配失败！\n");
        exit(EXIT_FAILURE);
    }
    P->next = NULL;
}

polynomial initList4c(void){
    polynomial P = (LinkList)malloc(sizeof(LNode));
    if (!P) {
        printf("内存分配失败！\n");
        exit(EXIT_FAILURE);
    }
    P->next = NULL;
    return P;
}

//获取头结点
LinkList GetHead(polynomial P){
    return P;
}
//设置当前结点的数据域
void SetCurElem(LinkList p, ElemType e){
    p->data = e;
}

//比较大小，a的指数值< = > b的指数值，分别返回-1，0，+1
int cmp(ElemType a, ElemType b){
    if(a.expn < b.expn) return -1;
    else if(a.expn == b.expn) return 0;
    else return 1;
}

//定位元素
Status LocateElem(LinkList L, ElemType e, Position &q, int (*cmp)(ElemType, ElemType)){
    q = L;
    Position p = L ? L->next : NULL;
    if (!p) return FALSE;
    while(p && cmp(p->data, e) < 0){
        q = p;
        p = p->next;
    }
    if (p && cmp(p->data, e) == 0){
        return TRUE;
    }
    return FALSE;

}

//创建新结点
Status MakeNode(LinkList *s, ElemType e){
    *s = (LinkList)malloc(sizeof(LNode));
    if (!*s) return FALSE;
    (*s)->data = e;
    (*s)->next = NULL;
    return TRUE;
}

//生成的结点插入链表
Status InsFirst(Position q, LinkList s){
    s->next = q->next;
    q->next = s;
    return TRUE;
}

//获取下一个结点位置
Position NextPos(polynomial P, Position p){
    return p->next;
}

//获取当前结点的数据域
ElemType GetCurElem(Position p){
    return p->data;
}

//删除结点
Status DelFirst(Position &h, Position q){
    h->next=q->next;
    return TRUE;
}

//释放结点
Status FreeNode(Position &p){
    free(p);
    p = NULL;
    return TRUE;
}    

//判断链表是否为空
Status ListEmpty(polynomial P){
    return P->next == NULL;
}
//尾部追加链表
Status Append(polynomial &P, Position q){
    Position p = P;
    while(p->next){
        p = p->next;
    }
    p->next = q;
    return TRUE;
}



//1.输入并创建多项式，指数升序排列
/*void Creatpolyn(polynomial &P, int n){
    //输入m项的系数和指数，建立表示一元多项式的有序链表p
    initList(P);
    Position h=GetHead(P);
    //设置头节点
    h->data.coef = 0.0f;
    h->data.expn = n; //多项式的项数

    if(n <= 0) return;

    for(int i=0;i<n;i++){
        ElemType e;
        printf("请输入第%d项的系数和指数：\n",i+1);
        if(scanf("%f %d",&e.coef,&e.expn)!=2){
            printf("输入格式错误！\n");
            exit(EXIT_FAILURE);
        }
        Position q;
        if(!LocateElem(h,e,q,(*cmp))){
            LinkList s = NULL;
            if(MakeNode(&s,e)) InsFirst(q,s);
            else{
                printf("内存分配失败！\n");
                exit(EXIT_FAILURE);
            }
        }
        else{
            //已存在，合并同类项
            Position cur = q->next;
            cur->data.coef += e.coef;
            if(fabs(cur->data.coef) < EPS){
                //系数为0，删除该结点
                DelFirst(q,cur);
                FreeNode(cur);
                h->data.expn -= 1; //多项式项数减1
            }
        }
    }

}//Createpolyn*/
void Creatpolyn(polynomial &P, int n) {
    // 输入 m 项的系数和指数，建立表示一元多项式的有序链表 P
    initList(P);
    Position h = GetHead(P);
    // 设置头节点
    h->data.coef = 0.0f;
    h->data.expn = n; // 多项式的项数

    if (n <= 0) return;

    for (int i = 0; i < n; i++) {
        ElemType e;
        printf("请输入第%d项的系数和指数：\n", i + 1);
        if (scanf("%f %d", &e.coef, &e.expn) != 2) {
            printf("输入格式错误！请重新运行程序。\n");
            exit(EXIT_FAILURE);
        }
        Position q;
        if (!LocateElem(h, e, q, (*cmp))) {
            LinkList s = NULL;
            if (MakeNode(&s, e)) {
                InsFirst(q, s);
            } else {
                printf("内存分配失败！\n");
                exit(EXIT_FAILURE);
            }
        } else {
            // 已存在，合并同类项
            Position cur = q->next;
            cur->data.coef += e.coef;
            if (fabs(cur->data.coef) < EPS) {
                // 系数为 0，删除该结点
                DelFirst(q, cur);
                FreeNode(cur);
                h->data.expn -= 1; // 多项式项数减 1
            }
        }
    }
}


//2.输出多项式
void PrintPolyn(polynomial P){
    if(!P){
        printf("多项式不存在！\n");
        return;
    }
    if (!P->next){
        printf("0 0\n");
        return;
    }
    Position p = P;
    printf("%d ", (int)(P->data.expn)); //输出多项式的项数
    p = p->next;
    while(p){
        if(fabs(p->data.coef) < EPS){
            p = p->next;
            continue;
        }
        printf("%.1f %d ", p->data.coef, p->data.expn);
        p = p->next;
    }
    printf("\n");
}

//3. 求和
void Addpolyn(polynomial &Pa, polynomial &Pb){
    //多项式加法，利用两个多项式的结点构成“和多项式”,合并到
    LinkList ha = GetHead(Pa);
    LinkList hb = GetHead(Pb);
    LinkList qa = NextPos(Pa,ha);
    LinkList qb = NextPos(Pb,hb);
    Position h = GetHead(Pa);
    h->data.expn = 0; //和多项式的项数初始化为0
    while(qa && qb){
        ElemType a = GetCurElem(qa);
        ElemType b = GetCurElem(qb);
        switch(cmp(a,b)){
            case -1: //多项式PA中当前结点指数值小
                ha = qa;
                qa=NextPos(Pa,qa);
                h->data.expn += 1;
                break;
            case 0: //指数值相等
                ElemType sum;
                sum.coef = a.coef + b.coef;
                sum.expn = a.expn;
                if(sum.coef != 0.0){
                    SetCurElem(qa,sum);
                    ha=qa;
                }
                else{
                    //删除多项式PA中当前结点
                    DelFirst(ha,qa);
                    FreeNode(qa);
                }
                DelFirst(hb,qb);
                FreeNode(qb);
                qb = NextPos(Pb,hb);
                qa = NextPos(Pa, ha);
                h->data.expn += 1;
                break;
            case 1: //多项式PB中当前结点指数值小
                DelFirst(hb,qb);
                InsFirst(ha,qb);
                qb = NextPos(Pb,hb);
                ha = NextPos(Pa,ha);
                h->data.expn += 1;
                break;
        }//switch
    }//while
    if(!ListEmpty(Pb)){
        Append(Pa, qb);
        h->data.expn += hb->data.expn;
    } 
}//Addpolyn

//4. 求差
void SubPolyn(polynomial &Pa, polynomial &Pb){

    LinkList ha = GetHead(Pa);
    LinkList hb = GetHead(Pb);
    LinkList qa = NextPos(Pa,ha);
    LinkList qb = NextPos(Pb,hb);
    Position h = GetHead(Pa);
    h->data.expn = 0; //和多项式的项数初始化为0
    while(qa && qb){
        ElemType a = GetCurElem(qa);
        ElemType b = GetCurElem(qb);
        switch(cmp(a,b)){
            case -1: //多项式PA中当前结点指数值小
                ha = qa;
                qa=NextPos(Pa,qa);
                h->data.expn += 1;
                break;
            case 0: //指数值相等
                ElemType sum;
                sum.coef = a.coef - b.coef;
                sum.expn = a.expn;
                if(sum.coef != 0.0){
                    SetCurElem(qa,sum);
                    ha=qa;
                }
                else{
                    //删除多项式PA中当前结点
                    DelFirst(ha,qa);
                    FreeNode(qa);
                }
                DelFirst(hb,qb);
                FreeNode(qb);
                qb = NextPos(Pb,hb);
                qa = NextPos(Pa, ha);
                h->data.expn += 1;
                break;
            case 1: //多项式PB中当前结点指数值小
                DelFirst(hb,qb);
                b = GetCurElem(qb);
                b.coef = -b.coef; //取相反数
                SetCurElem(qb,b);
                InsFirst(ha,qb);
                qb = NextPos(Pb,hb);
                ha = NextPos(Pa,ha);
                h->data.expn += 1;
                break;
        }//switch
    }//while
    if(!ListEmpty(Pb)){
        Position p = qb;
        while(p){
            ElemType b = GetCurElem(p);
            b.coef = -b.coef; //取相反数
            SetCurElem(p,b);
            p = NextPos(Pb,p);
        }
        Append(Pa, qb);
    }
    FreeNode(hb);
}

//5.求值
float EvalPolyn(polynomial P, float x){
    float result = 0.0f;
    Position p;
    p = NextPos(P, GetHead(P));
    while(p){
        result += p->data.coef * powf(x, p->data.expn);
        p = NextPos(P, p);
    }
    return result;
}

//6. 销毁
void DestroyPolyn(polynomial &P){
    Position p = GetHead(P);
    Position q;
    while(p){
        q = NextPos(P,p);
        FreeNode(p);
        p = q;
    }
}

//7. 清空
void ClearPolyn(polynomial &P){
    Position p  = GetHead(P);
    Position q = NextPos(P,p);
    while(q){
        DelFirst(p,q);
        FreeNode(q);
        q = NextPos(P,p);
    }
    p->data.coef = 0.0f;
    p->data.expn = 0;
}

//8. 修改
//8.1 插入新结点
Status InsNewNode(polynomial &P, ElemType e){
    Position h = GetHead(P);
    Position p = GetHead(P);
    if(!p) return FALSE;
    Position q = NextPos(P,p);
    while(q && e.expn > q->data.expn){
        p = q;
        q = NextPos(P,q);
    }
    if(q && e.expn == q->data.expn){
        //合并同类项
        q->data.coef += e.coef;
        if(fabs(q->data.coef) < EPS){
            //删除结点
            p->next = NextPos(P,q);
            FreeNode(q);
            h->data.expn -= 1;
        }//if
        return TRUE;
    }//if
    else{
        //插入新结点到p和q之间
        LinkList s = (LinkList)malloc(sizeof(LNode));
        if(!s) return FALSE;
        s->data = e;
        s->next = q;
        p->next = s;
        h->data.expn += 1;
        return TRUE;
    }
}

//8.2 删除结点
Status DelNode(polynomial &P, int expn){
    Position p = GetHead(P);
    Position q = NextPos(P,p);
    if(!q) return FALSE;
    while(q && q->data.expn != expn){
        p = q;
        q = NextPos(P,q);
    }
    if(!q) return FALSE;
    else{
        p->next = NextPos(P,q);
        FreeNode(q);
        return TRUE;
    }
}

//8.3 修改已有结点的系数和指数
Status ChangeNode(polynomial &P, ElemType e){

    Position q = NextPos(P,GetHead(P));
    while(q && q->data.expn != e.expn){
        q = NextPos(P,q);
    }
    if(!q) return FALSE;
    else{
        if(e.coef == 0.0f){
            DelNode(P, e.expn);
        }
        q->data.coef = e.coef;
        return TRUE;
    }
}

//拓展功能
//9. 微分
void DiffPolyn(polynomial &P, int N){
    if (N <= 0 || !P) return;
    Position p = GetHead(P);
    Position q = NextPos(P,p);
    int i = 0;
    int k = 0;
    while(q){
        i=0;
        if(q->data.expn >= 0 && q->data.expn < N){
            DelFirst(p,q);
            FreeNode(q);
            q = NextPos(P, p);
            k++;
        }//if
        else{
            while(i<N){
                q->data.coef *= q->data.expn;
                q->data.expn -= 1;
                i++;
            }//while
            p = q;
            q=NextPos(P,q);
        }//else
    }//while
    PrintPolyn(P);
}

//10. 不定积分
void IndIntegPolyn(polynomial &P){
    Position p = GetHead(P);
    Position q = NextPos(P,p);
    while(q){
        if(q->data.expn == -1){
            p = q;
            q = NextPos(P,q);
            continue;
        }
        q->data.expn += 1;
        q->data.coef /= q->data.expn;
        p=q;
        q=NextPos(P,q);
    }
    p = GetHead(P);
    printf("%d", p->data.expn);
    p = NextPos(P,p);
    while(p){
        printf(" %.1f %d", p->data.coef, p->data.expn);
        p = NextPos(P,p);
    }
}

//12. 乘法和乘方
polynomial MultiPolyn(polynomial Pa, polynomial Pb){
    polynomial Pc=initList4c();
    if(!Pa || !Pb) return NULL;

    for(Position p=NextPos(Pa, GetHead(Pa));p;p=NextPos(Pa,p)){
        for (Position q=NextPos(Pb, GetHead(Pb));q;q=NextPos(Pb,q)){
            ElemType e;
            e.coef = p->data.coef * q->data.coef;
            e.expn = p->data.expn + q->data.expn;
            InsNewNode(Pc,e);
        }//for
    }//for
    return Pc;
}
/*polynomial PowPolyn(polynomial Pa, int k){
    polynomial Pb,Pc;
    initList(Pb);
    initList(Pc);
    if(k<=0){
        ElemType e;
        e.coef = 1.0f;
        e.expn = 0;
        InsNewNode(Pb,e);
        return Pb;
    }
    else if(k==1){
        return Pa;
    }
    else{
        Pb = Pa;
        for (int i=0;i<k;i++){
            Pc = MultiPolyn(Pb,Pa);
            DestroyPolyn(Pb);
            Pb = Pc;
        }
        return Pb;
    }
}*/
polynomial ClonePolyn(polynomial P){
    polynomial C = initList4c();       // 创建新头
    for (Position p = NextPos(P, GetHead(P)); p; p = NextPos(P, p)){
        InsNewNode(C, p->data);        // 逐项拷贝
    }
    // 维护项数（可选：InsNewNode 已经维护的话可以省）
    C->data.expn = P->data.expn;
    return C;
}

polynomial PowPolyn(polynomial Pa, int k){
    // R = 1
    polynomial R = initList4c();
    ElemType one;
    one.coef = 1.0f;
    one.expn = 0;

    InsNewNode(R, one);

    if (k <= 0) return R;
    // base = Pa 的副本，保护原多项式
    polynomial base = ClonePolyn(Pa);

    while (k > 0){
        if (k & 1){
            polynomial T = MultiPolyn(R, base);
            DestroyPolyn(R);
            R = T;
        }
        k >>= 1;
        if (k > 0){ // 最后一次不必再平方
            polynomial B2 = MultiPolyn(base, base);
            DestroyPolyn(base);
            base = B2;
        }
    }
    DestroyPolyn(base); // 收尾清理
    return R;
}


//多个多项式头指针数组存储
//初始化函数
void InitPolynArray(){
    for(int i=0;i<MAX_POLY_NUM;++i){
        initList(P[i]);
    }
}   



//主函数
int main(){
    InitPolynArray();
    int m, n ,i,j,k;
    float x;
    m=1;
    i=1;
    printf("请选择操作：\n");
    printf("1. 输入并创建多项式\n2. 输出多项式\n3. 求和\n4. 求差\n5. 求值\n6. 销毁\n7. 清空\n8. 修改\n9.微分\n10.不定积分\n12.乘法和乘方\n（输入0结束操作）");
    while(m!=0 && i<=MAX_POLY_NUM){
        scanf("%d", &m);
        switch(m){
            case 1:{
                printf("请选择要创建的多项式的编号:");
                scanf("%d", &j);
                if(P[j]->next != NULL){
                    printf("多项式%d已存在，无法创建！\n",j);
                    break;
                }
                printf("请输入多项式%d的项数：\n",j);
                scanf("%d", &n);
                Creatpolyn(P[j], n);
                if(j>i)i++;
                break;
            }

            case 2:{
                printf("请选择打印第几个多项式：（目前有%d个）",i);
                scanf("%d", &k);
                PrintPolyn(P[k]);
                break;
            }
            case 3:{
                printf("选择要求和的两个多项式：(目前有%d个)",i);
                scanf("%d %d", &j, &k);
                Addpolyn(P[j], P[k]);
                PrintPolyn(P[j]);
                break;
            }
            case 4:{
                printf("选择要求差的两个多项式：(目前有%d个)",i);
                scanf("%d %d", &j, &k);
                SubPolyn(P[j], P[k]);
                PrintPolyn(P[j]);
                break;
            }
            case 5:{
                printf("选择要求值的多项式：(目前有%d个)",i);
                scanf("%d", &j);
                printf("请输入x的值：\n");
                scanf("%f", &x);
                float result = EvalPolyn(P[j], x);
                printf("多项式在x=%f时的值为：%.2f\n", x, result);
                break;
            }
            case 6:{
                printf("选择要销毁的多项式：(目前有%d个)",i);
                scanf("%d", &j);
                DestroyPolyn(P[j]);
                break;
            }
            case 7:{
                printf("选择要清空的多项式：(目前有%d个)",i);
                scanf("%d", &j);
                ClearPolyn(P[j]);
                break;
            }
            case 8:{
                printf("请选择要修改的多项式：(目前有%d个)",i);
                scanf("%d", &j);
                printf("请选择修改方式：1.插入新结点 2.删除结点 3.修改已有结点");
                int choice;
                scanf("%d", &choice);
                switch(choice){
                    case 1:{
                        ElemType e;
                        printf("请输入要插入的结点的系数和指数：\n");
                        scanf("%f %d", &e.coef, &e.expn);
                        InsNewNode(P[j], e);
                        break;
                    }
                    case 2:{
                        int expn;
                        printf("请输入要删除的结点的指数：\n");
                        scanf("%d", &expn);
                        DelNode(P[j], expn);
                        break;
                    }
                    case 3:{
                        ElemType e2;
                        printf("请输入要修改的结点的指数和新的系数：\n");
                        scanf("%d %f", &e2.expn, &e2.coef);
                        ChangeNode(P[j], e2);
                        break;
                    }
                }
                break;
            }
            case 9: {
                printf("选择要微分的多项式：(目前有%d个)",i);
                scanf("%d", &j);
                printf("请输入微分的阶数：\n");
                scanf("%d", &k);
                DiffPolyn(P[j], k);
                break;
            }
            case 10:{
                printf("选择要不定积分的多项式：(目前有%d个)",i);
                scanf("%d", &j);
                IndIntegPolyn(P[j]);
                break;
            }
            case 12:{
                int choice;
                printf("请选择要进行乘法还是乘方操作：1.乘法 2.乘方\n");
                scanf("%d", &choice);
                if(choice == 1){
                    printf("选择要相乘的两个多项式：(目前有%d个)",i);
                    scanf("%d %d",&j, &k);
                    polynomial Pc = MultiPolyn(P[j], P[k]);
                    PrintPolyn(Pc);
                }
                else if(choice == 2){
                    printf("请选择要进行乘方的多项式：(目前有%d个)",i);
                    scanf("%d",&j);
                    printf("请输入乘方次数：\n");
                    scanf("%d",&k);
                    polynomial Pd = PowPolyn(P[j], k);
                    PrintPolyn(Pd);
                }
                break;
            }
            case 0:
                break;
        }//switch
        printf("请选择操作：\n");
        printf("1. 输入并创建多项式\n2. 输出多项式\n3. 求和\n4. 求差\n5. 求值\n6. 销毁\n7. 清空\n8. 修改\n9.微分\n10.不定积分\n12.乘法和乘方\n（输入0结束操作）");
    }//while
}




typedef struct {
    float coef; // 系数
    int expn;   // 指数
} ElemType;

typedef struct node {
    ElemType data;      // 数据域
    struct node *next;  // 指针域
} LNode, *LinkList;

polynomial P[MAX_POLY_NUM]